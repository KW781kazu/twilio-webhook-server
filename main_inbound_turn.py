# =========================
# main_inbound_turn.py  (FULL / Flask-Sock + eventlet)
# =========================
import os
import sys
import json
import time
import base64
import logging
from typing import Optional

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

# WebSocket
from flask_sock import Sock

# OpenAI (optional)
try:
    from openai import OpenAI
    _openai_enabled = True
except Exception:
    _openai_enabled = False

# AWS Polly (optional)
try:
    import boto3
    _polly_enabled = True
except Exception:
    _polly_enabled = False


# ============== Settings ==============
APP_NAME = "twilio-media-whisper-gpt"
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
POLLY_VOICE = os.environ.get("POLLY_VOICE", "Mizuki")
SAMPLE_RATE = 8000  # Twilio Media Streams spec

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(APP_NAME)

app = Flask(__name__)
sock = Sock(app)


# ============== Utils ==============
def xml_response(xml_str: str) -> Response:
    return Response(xml_str, mimetype="text/xml; charset=utf-8")

def is_secure_base() -> bool:
    return PUBLIC_BASE.startswith("https://")

def ws_absolute_url(path: str) -> str:
    host = PUBLIC_BASE.replace("https://", "").replace("http://", "")
    scheme = "wss" if is_secure_base() else "ws"
    return f"{scheme}://{host}{path}"


# ============== Health/Version ==============
@app.route("/healthz", methods=["GET"])
def healthz() -> Response:
    return Response("ok", mimetype="text/plain")

@app.route("/version", methods=["GET"])
def version() -> Response:
    info = {
        "app": APP_NAME,
        "ws_url": ws_absolute_url("/media"),
        "public_base": PUBLIC_BASE,
        "openai_enabled": _openai_enabled,
        "polly_enabled": _polly_enabled,
    }
    return Response(json.dumps(info), mimetype="application/json")


# ============== TwiML (GET/POST) ==============
def build_twiml() -> str:
    vr = VoiceResponse()
    vr.say("こんにちは。こちらは受付です。ピーという音のあとにご用件をお話しください。", language="ja-JP")

    # Media Streams（inbound）
    stream_url = ws_absolute_url("/media")
    log.info(f"[TwiML] stream url -> {stream_url}")
    with vr.connect() as c:
        c.stream(url=stream_url, track="inbound")  # 双方向にするなら bidirectional="true"
    return str(vr)

@app.route("/voice", methods=["GET", "POST"])
def voice() -> Response:
    return xml_response(build_twiml())


# ============== Media helpers (placeholders) ==============
def decode_twilio_ulaw_b64_to_pcm(b64_payload: str) -> bytes:
    try:
        return base64.b64decode(b64_payload)
    except Exception:
        return b""

def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    if not ulaw_bytes:
        return b""
    if not hasattr(ulaw_to_pcm16, "_table"):
        table = []
        for i in range(256):
            u = ~i & 0xFF
            sign = (u & 0x80)
            exponent = (u >> 4) & 0x07
            mantissa = u & 0x0F
            magnitude = ((mantissa << 3) + 0x84) << exponent
            sample = magnitude - 0x84
            sample = -sample if sign else sample
            sample = max(-32768, min(32767, sample))
            table.append(sample & 0xFFFF)
        ulaw_to_pcm16._table = table
    out = bytearray()
    for b in ulaw_bytes:
        s = ulaw_to_pcm16._table[b]
        out.append(s & 0xFF); out.append((s >> 8) & 0xFF)
    return bytes(out)

def whisper_transcribe_pcm16(pcm16_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> str:
    if not _openai_enabled or not OPENAI_API_KEY:
        return ""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return ""
    except Exception as e:
        log.warning(f"[ASR] Whisper error: {e}")
        return ""

def chat_generate(prompt: str, system: str = "あなたは丁寧な日本語の電話受付AIです。") -> str:
    if not _openai_enabled or not OPENAI_API_KEY:
        return ""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"[NLG] Chat error: {e}")
        return ""

def polly_tts_to_ulaw_b64(text: str) -> Optional[str]:
    if not _polly_enabled:
        return None
    try:
        polly = boto3.client("polly", region_name=AWS_REGION)
        _ = polly.synthesize_speech(
            Text=text, OutputFormat="pcm", SampleRate=str(SAMPLE_RATE),
            VoiceId=POLLY_VOICE, LanguageCode="ja-JP"
        )
        return None
    except Exception as e:
        log.warning(f"[TTS] Polly error: {e}")
        return None


# ============== Simple VAD placeholder ==============
class SimpleTurnDetector:
    def __init__(self, idle_ms: int = 900):
        self.idle_ms = idle_ms
        self.last_audio_ts = time.time()
    def on_audio(self):
        self.last_audio_ts = time.time()
    def is_silence_gap(self) -> bool:
        return (time.time() - self.last_audio_ts) * 1000.0 > self.idle_ms


# ============== WebSocket: /media ==============
@sock.route("/media")
def media(ws):
    # Render のプロキシ越しでも X-Forwarded-For が入る
    remote = request.headers.get("X-Forwarded-For") or request.remote_addr
    log.info(f"[WS] upgrade request from {remote}")
    log.info("[WS] connected (handshake OK)")

    call_sid = None
    detector = SimpleTurnDetector(idle_ms=900)

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                log.info("[WS] client closed")
                break
            try:
                data = json.loads(raw)
            except Exception:
                log.debug("[WS] non-JSON frame ignored")
                continue

            event = data.get("event")
            if event == "start":
                call_sid = data.get("start", {}).get("callSid")
                stream_sid = data.get("start", {}).get("streamSid")
                log.info(f"[STATUS] stream-started callSid={call_sid} streamSid={stream_sid}")

            elif event == "media":
                payload = data.get("media", {}).get("payload", "")
                _ = decode_twilio_ulaw_b64_to_pcm(payload)
                detector.on_audio()

            elif event == "mark":
                mark = data.get("mark", {}).get("name")
                log.debug(f"[WS] mark: {mark}")

            elif event == "stop":
                log.info("[STATUS] stream-stopped callSid=%s", call_sid)
                break

            else:
                log.debug(f"[WS] event: {event}")

    except Exception as e:
        log.warning(f"[WS] exception: {e}")
    finally:
        log.info("[WS] closed callSid=%s", call_sid)


# ============== Local dev ==============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
