# =========================
# main_inbound_turn.py  (FULL)
# =========================
import os
import sys
import json
import time
import base64
import logging
import queue
import threading
from typing import Optional, Tuple, List

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

# --- WebSocket (gevent) ---
# Gunicorn worker: geventwebsocket.gunicorn.workers.GeventWebSocketWorker
# pip: flask-sockets, gevent, gevent-websocket
from flask_sockets import Sockets
from gevent import queue as gevent_queue

# --- (Optional) OpenAI Whisper / GPT ---
# pip: openai>=1.30
try:
    from openai import OpenAI
    _openai_enabled = True
except Exception:
    _openai_enabled = False

# --- (Optional) AWS Polly TTS ---
# pip: boto3
try:
    import boto3
    _polly_enabled = True
except Exception:
    _polly_enabled = False


# =========================
# Settings
# =========================
APP_NAME = "twilio-media-whisper-gpt"
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
POLLY_VOICE = os.environ.get("POLLY_VOICE", "Mizuki")

# Twilio Media Streams audio spec
SAMPLE_RATE = 8000  # Hz
CHANNELS = 1
# Twilio sends μ-law (PCMU) 8kHz mono, base64 in "media.payload"

# Logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(APP_NAME)

app = Flask(__name__)
sockets = Sockets(app)


# =========================
# Utility
# =========================
def xml_response(xml_str: str) -> Response:
    return Response(xml_str, mimetype="text/xml; charset=utf-8")


def is_secure_base() -> bool:
    return PUBLIC_BASE.startswith("https://")


def ws_absolute_url(path: str) -> str:
    # Twilio requires wss
    host = PUBLIC_BASE.replace("https://", "").replace("http://", "")
    scheme = "wss" if is_secure_base() else "ws"
    return f"{scheme}://{host}{path}"


# =========================
# Health / Version
# =========================
@app.route("/healthz", methods=["GET"])
def healthz() -> Response:
    return Response("ok", mimetype="text/plain")


@app.route("/version", methods=["GET"])
def version() -> Response:
    info = {
        "app": APP_NAME,
        "ws_path": "/media",
        "public_base": PUBLIC_BASE,
        "openai_enabled": _openai_enabled,
        "polly_enabled": _polly_enabled,
    }
    return Response(json.dumps(info), mimetype="application/json")


# =========================
# TwiML (GET/POST both OK)
# =========================
def build_twiml() -> str:
    vr = VoiceResponse()
    # 初期応答（Say）
    vr.say("こんにちは。こちらは受付です。ピーという音のあとにご用件をお話しください。", language="ja-JP")

    # Media Streams（inbound。双方向にしたい場合は bidirectional=\"true\" を追加）
    stream_url = ws_absolute_url("/media")
    with vr.connect() as c:
        c.stream(url=stream_url, track="inbound")  # bidirectional="true" も可

    return str(vr)


@app.route("/voice", methods=["GET", "POST"])
def voice() -> Response:
    # GET/POST いずれでも同じ TwiML を返す（デバッグ用）
    xml = build_twiml()
    return xml_response(xml)


# =========================
# Audio / ASR / NLG / TTS (placeholders)
# =========================
def decode_twilio_ulaw_b64_to_pcm(b64_payload: str) -> bytes:
    """
    Twilio sends base64 μ-law frames. Here we simply return raw bytes (μ-law).
    If your ASR expects linear PCM, you must convert μ-law -> PCM16 8kHz.
    """
    try:
        return base64.b64decode(b64_payload)
    except Exception:
        return b""


def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """
    μ-law -> PCM16 conversion (lookup-table based).
    To keep this file self-contained, we implement a simple converter.
    """
    if not ulaw_bytes:
        return b""

    # μ-law decode table
    # Precompute only once
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
            # clamp to 16-bit
            if sample > 32767:
                sample = 32767
            if sample < -32768:
                sample = -32768
            table.append(sample & 0xFFFF)
        ulaw_to_pcm16._table = table

    out = bytearray()
    tbl = ulaw_to_pcm16._table
    for b in ulaw_bytes:
        s = tbl[b]
        out.append(s & 0xFF)
        out.append((s >> 8) & 0xFF)
    return bytes(out)


def whisper_transcribe_pcm16(pcm16_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Send audio chunk to Whisper (OpenAI). Keep short to reduce latency.
    You can replace with your own streaming ASR.
    """
    if not _openai_enabled or not OPENAI_API_KEY:
        return ""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # 真のストリーミングは別途実装。ここでは安定運用を優先し空文字を返す。
        return ""
    except Exception as e:
        log.warning(f"[ASR] Whisper error: {e}")
        return ""


def chat_generate(prompt: str, system: str = "あなたは丁寧な日本語の電話受付AIです。") -> str:
    """
    Call GPT-4o-mini (または任意モデル)。
    """
    if not _openai_enabled or not OPENAI_API_KEY:
        return ""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"[NLG] Chat error: {e}")
        return ""


def polly_tts_to_ulaw_b64(text: str) -> Optional[str]:
    """
    Polly TTS -> PCM16 -> μ-law -> base64（戻し送信する場合に使用）。
    いまは握手安定を優先し None を返す。
    """
    if not _polly_enabled:
        return None

    try:
        polly = boto3.client("polly", region_name=AWS_REGION)
        synth = polly.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            SampleRate=str(SAMPLE_RATE),
            VoiceId=POLLY_VOICE,
            LanguageCode="ja-JP",
        )
        pcm16 = synth["AudioStream"].read()
        # ここで μ-law 変換して返す実装を追加可能
        return None
    except Exception as e:
        log.warning(f"[TTS] Polly error: {e}")
        return None


# =========================
# Simple VAD (timer-based placeholder)
# =========================
class SimpleTurnDetector:
    def __init__(self, idle_ms: int = 900):
        self.idle_ms = idle_ms
        self.last_audio_ts = time.time()

    def on_audio(self):
        self.last_audio_ts = time.time()

    def is_silence_gap(self) -> bool:
        return (time.time() - self.last_audio_ts) * 1000.0 > self.idle_ms


# =========================
# WebSocket: /media
# =========================
@sockets.route("/media")
def media_socket(ws):
    """
    Twilio <Stream> がここに wss 接続します。
    JSON テキストフレーム（event=start/media/stop）を受信します。
    """
    remote = request.environ.get("HTTP_X_FORWARDED_FOR", request.remote_addr)
    call_sid = None
    log.info("[WS] new connection request from %s", remote)

    send_q = gevent_queue.Queue()
    detector = SimpleTurnDetector(idle_ms=900)

    def sender():
        try:
            while not ws.closed:
                msg = send_q.get()
                if msg is None:
                    break
                ws.send(msg)
        except Exception as e:
            log.warning(f"[WS] sender exception: {e}")

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    try:
        log.info("[WS] connected (handshake OK)")
        while not ws.closed:
            raw = ws.receive()
            if raw is None:
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
                media = data.get("media", {})
                payload = media.get("payload", "")
                ulaw_bytes = decode_twilio_ulaw_b64_to_pcm(payload)
                detector.on_audio()
                # pcm16 = ulaw_to_pcm16(ulaw_bytes)  # ASR するなら使用
                pass

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
        try:
            send_q.put(None)
        except Exception:
            pass
        log.info("[WS] closed callSid=%s", call_sid)


# =========================
# Local dev entrypoint
# (Gunicorn will import app object)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
