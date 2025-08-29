# =========================
# main_inbound_turn.py  (FULL / eventlet + flask-sock + one-way response via Calls API)
# =========================
import os
import sys
import json
import time
import base64
import wave
import uuid
import logging
import threading
from typing import Optional, Dict, List

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from flask_sock import Sock

# Twilio REST (通話中にTwiMLを更新するために使用)
from twilio.rest import Client as TwilioClient

# OpenAI (Whisper / GPT)
try:
    from openai import OpenAI
    _openai_enabled = True
except Exception:
    _openai_enabled = False

# Polly は未使用（片方向なので再生は <Say>）
_polly_enabled = False


# ===== Settings =====
APP_NAME = "twilio-media-whisper-gpt"
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

SAMPLE_RATE = 8000          # Twilio Media Streams は 8kHz μ-law
VAD_IDLE_MS = 1000          # 無音判定（ms）
MIN_UTTER_MS = 800          # 認識に投げる最小発話長（ms）

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(APP_NAME)

app = Flask(__name__)
sock = Sock(app)

# Twilio REST client
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# OpenAI client
oai_client = OpenAI(api_key=OPENAI_API_KEY) if _openai_enabled and OPENAI_API_KEY else None


# ===== Utilities =====
def xml_response(xml_str: str) -> Response:
    return Response(xml_str, mimetype="text/xml; charset=utf-8")

def is_secure_base() -> bool:
    return PUBLIC_BASE.startswith("https://")

def ws_absolute_url(path: str) -> str:
    host = PUBLIC_BASE.replace("https://", "").replace("http://", "")
    scheme = "wss" if is_secure_base() else "ws"
    return f"{scheme}://{host}{path}"

def build_reconnect_twiml(say_text: str) -> str:
    """
    <Say> で応答再生 → <Connect><Stream .../></Connect> で WS 再接続
    """
    vr = VoiceResponse()
    if say_text.strip():
        vr.say(say_text, language="ja-JP")
    with vr.connect() as c:
        c.stream(url=ws_absolute_url("/media"))  # track は省略（デフォルトで inbound）
    return str(vr)


# ===== Health / Version =====
@app.route("/healthz", methods=["GET"])
def healthz() -> Response:
    return Response("ok", mimetype="text/plain")

@app.route("/version", methods=["GET"])
def version() -> Response:
    info = {
        "app": APP_NAME,
        "ws_url": ws_absolute_url("/media"),
        "public_base": PUBLIC_BASE,
        "openai_enabled": bool(oai_client),
    }
    return Response(json.dumps(info), mimetype="application/json")


# ===== TwiML (初期応答＋ストリーム開始) =====
def build_initial_twiml() -> str:
    vr = VoiceResponse()
    vr.say("こんにちは。こちらは受付です。ピーのあとにご用件をお話しください。", language="ja-JP")
    stream_url = ws_absolute_url("/media")
    log.info(f"[TwiML] stream url -> {stream_url}")
    with vr.connect() as c:
        c.stream(url=stream_url)
    return str(vr)

@app.route("/voice", methods=["GET", "POST"])
def voice() -> Response:
    return xml_response(build_initial_twiml())


# ===== /media の HTTP到達確認（WS以外のGETでも200） =====
@app.route("/media", methods=["GET"])
def media_http_probe() -> Response:
    return Response("media endpoint (HTTP) is alive", mimetype="text/plain")


# ===== μ-law utilities =====
def decode_ulaw_b64(b64_payload: str) -> bytes:
    try:
        return base64.b64decode(b64_payload)
    except Exception:
        return b""

# μ-law -> PCM16 (8kHz mono)
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

def write_wav_8k_pcm16(pcm16_bytes: bytes, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16_bytes)


# ===== Very simple VAD =====
class SimpleVAD:
    def __init__(self, idle_ms: int):
        self.idle_ms = idle_ms
        self.last_ts = time.time()

    def on_audio(self):
        self.last_ts = time.time()

    def silence_ms(self) -> float:
        return (time.time() - self.last_ts) * 1000.0

    def is_end_of_utterance(self) -> bool:
        return self.silence_ms() > self.idle_ms


# ===== Per-call buffer =====
class CallBuffer:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.ulaw_chunks: List[bytes] = []
        self.started_at = time.time()

    def append_ulaw(self, b: bytes):
        if b:
            self.ulaw_chunks.append(b)

    def reset(self):
        self.ulaw_chunks.clear()

    def total_ms(self) -> int:
        # μ-law 8kHz mono: 8000 bytes ≒ 1秒（おおよそ）
        total = sum(len(c) for c in self.ulaw_chunks)
        return int(total / 8)  # 8000 bytes ≒ 1000ms → /8 でms換算近似

    def export_wav_path(self) -> Optional[str]:
        if not self.ulaw_chunks:
            return None
        pcm16 = ulaw_to_pcm16(b"".join(self.ulaw_chunks))
        path = f"/tmp/{self.call_sid}_{uuid.uuid4().hex}.wav"
        write_wav_8k_pcm16(pcm16, path)
        return path


# ===== State =====
call_buffers: Dict[str, CallBuffer] = {}
call_vads: Dict[str, SimpleVAD] = {}


# ===== Speech pipeline =====
def run_pipeline_and_reply(call_sid: str, wav_path: Optional[str]):
    """
    Whisper -> GPT -> Twilio Calls API で <Say> 再生＆<Stream> 再接続
    """
    try:
        # 1) ASR
        text = ""
        if wav_path and oai_client:
            try:
                with open(wav_path, "rb") as f:
                    tr = oai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        # language="ja"  # Whisperは自動判定でもOK
                    )
                text = (tr.text or "").strip()
            except Exception as e:
                log.warning(f"[ASR] Whisper error: {e}")

        log.info(f"[ASR] text='{text}'")

        # 2) GPT 応答（空白なら聞き返し）
        reply = "恐れ入ります、もう一度ゆっくりお話しください。"
        if text:
            try:
                cr = oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "あなたは丁寧な日本語の電話受付AIです。簡潔に1〜2文で答えてください。"},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.4,
                )
                reply = cr.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"[NLG] Chat error: {e}")

        log.info(f"[NLG] reply='{reply}'")

        # 3) 通話中の TwiML を更新して再生 → 再接続
        twiml = build_reconnect_twiml(reply)
        twilio_client.calls(call_sid).update(twiml=twiml)
        log.info(f"[TwiML-UPDATE] replied and reconnected stream for callSid={call_sid}")

    finally:
        # 後片付け
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


# ===== WebSocket: /media =====
@app.before_request
def _log_incoming():
    if request.path == "/media":
        ua = request.headers.get("User-Agent", "")
        upg = request.headers.get("Upgrade", "")
        origin = request.headers.get("Origin", "")
        proto = request.headers.get("X-Forwarded-Proto", "")
        log.info(f"[PRE] path=/media method={request.method} "
                 f"upgrade={upg!r} origin={origin!r} xfwd_proto={proto!r} ua={ua!r}")

@sock.route("/media")
def media_ws(ws):
    call_sid = None
    try:
        log.info("[WS] connected (handshake OK)")
        while True:
            raw = ws.receive()
            if raw is None:
                log.info("[WS] client closed")
                break

            try:
                data = json.loads(raw)
            except Exception:
                continue

            event = data.get("event")
            if event == "start":
                call_sid = data.get("start", {}).get("callSid")
                stream_sid = data.get("start", {}).get("streamSid")
                log.info(f"[STATUS] stream-started callSid={call_sid} streamSid={stream_sid}")
                call_buffers[call_sid] = CallBuffer(call_sid)
                call_vads[call_sid] = SimpleVAD(VAD_IDLE_MS)

            elif event == "media" and call_sid:
                payload = data.get("media", {}).get("payload", "")
                ulaw = decode_ulaw_b64(payload)
                call_buffers[call_sid].append_ulaw(ulaw)
                call_vads[call_sid].on_audio()

                # VAD: 発話終了を検知したら処理スレッドへ
                if call_buffers[call_sid].total_ms() >= MIN_UTTER_MS and call_vads[call_sid].is_end_of_utterance():
                    buf = call_buffers[call_sid]
                    wav_path = buf.export_wav_path()
                    # 次の発話用にバッファをリセット
                    buf.reset()
                    # Whisper->GPT->TwiML更新を別スレで実行（通話を止めない）
                    threading.Thread(target=run_pipeline_and_reply, args=(call_sid, wav_path), daemon=True).start()

            elif event == "stop":
                log.info(f"[STATUS] stream-stopped callSid={call_sid}")
                break

    except Exception as e:
        log.warning(f"[WS] exception: {e}")
    finally:
        if call_sid:
            call_buffers.pop(call_sid, None)
            call_vads.pop(call_sid, None)
        log.info(f"[WS] closed callSid={call_sid}")


# ===== Local dev =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
