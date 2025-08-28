# =========================
# main_inbound_turn.py  (FULL / eventlet + flask-sock + handshake logging)
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
from flask_sock import Sock

try:
    from openai import OpenAI
    _openai_enabled = True
except Exception:
    _openai_enabled = False

try:
    import boto3
    _polly_enabled = True
except Exception:
    _polly_enabled = False


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


# ---------- utils ----------
def xml_response(xml_str: str) -> Response:
    return Response(xml_str, mimetype="text/xml; charset=utf-8")

def is_secure_base() -> bool:
    return PUBLIC_BASE.startswith("https://")

def ws_absolute_url(path: str) -> str:
    host = PUBLIC_BASE.replace("https://", "").replace("http://", "")
    scheme = "wss" if is_secure_base() else "ws"
    return f"{scheme}://{host}{path}"


# ---------- request logging (握手の可視化) ----------
@app.before_request
def _log_incoming():
    # WebSocket の Upgrade 要求や /media への到達を早期に可視化
    if request.path == "/media":
        ua = request.headers.get("User-Agent", "")
        upg = request.headers.get("Upgrade", "")
        orign = request.headers.get("Origin", "")
        proto = request.headers.get("X-Forwarded-Proto", "")
        log.info(f"[PRE] path=/media method={request.method} "
                 f"upgrade={upg!r} origin={orign!r} "
                 f"xfwd_proto={proto!r} ua={ua!r}")


# ---------- health/version ----------
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


# ---------- TwiML ----------
def build_twiml() -> str:
    vr = VoiceResponse()
    vr.say("こんにちは。こちらは受付です。ピーという音のあとにご用件をお話しください。", language="ja-JP")
    stream_url = ws_absolute_url("/media")
    log.info(f"[TwiML] stream url -> {stream_url}")
    with vr.connect() as c:
        c.stream(url=stream_url, track="inbound")
    return str(vr)

@app.route("/voice", methods=["GET", "POST"])
def voice() -> Response:
    return xml_response(build_twiml())


# ---------- HTTP の /media (到達確認用) ----------
@app.route("/media", methods=["GET"])
def media_http_probe() -> Response:
    # Twilio の WS ハンドシェイク以外（ただの GET 到達）でも 200 を返す
    # ※ WebSocket Upgrade の場合は下の @sock.route が処理する
    return Response("media endpoint (HTTP) is alive", mimetype="text/plain")


# ---------- WebSocket: /media ----------
@sock.route("/media")
def media_ws(ws):
    remote = request.headers.get("X-Forwarded-For") or request.remote_addr
    log.info(f"[WS] upgrade request from {remote}")
    log.info("[WS] connected (handshake OK)")

    call_sid = None

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
                # 音声フレーム受信（ここではペイロードを捨てる）
                pass

            elif event == "stop":
                log.info("[STATUS] stream-stopped callSid=%s", call_sid)
                break

            else:
                log.debug(f"[WS] event: {event}")

    except Exception as e:
        log.warning(f"[WS] exception: {e}")
    finally:
        log.info("[WS] closed callSid=%s", call_sid)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
