# =========================
# main_inbound_turn.py
# One-way: Twilio(Media Streams) -> Whisper -> GPT -> <Say> -> reconnect <Stream>
# 内容ベースのVAD（μ-law無音判定）で発話区切り
# =========================
import os, sys, json, time, base64, wave, uuid, logging, threading
from typing import Optional, Dict, List
from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient

# OpenAI (Whisper / GPT)
try:
    from openai import OpenAI
    _openai_enabled = True
except Exception:
    _openai_enabled = False

APP_NAME = "twilio-media-whisper-gpt"
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

SAMPLE_RATE = 8000
VAD_IDLE_MS = 900            # 有声→無声のギャップがこの時間を超えたら一区切り
MIN_UTTER_MS = 700           # 最低発話長（短すぎる断片を防ぐ）
SILENCE_RATIO = 0.90         # フレーム内の 0xFF 比率がこれ以上なら“無音”とみなす

logging.basicConfig(level=os.environ.get("LOG_LEVEL","INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(APP_NAME)

app = Flask(__name__)
sock = Sock(app)

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
oai_client = OpenAI(api_key=OPENAI_API_KEY) if (_openai_enabled and OPENAI_API_KEY) else None


# ---------- helpers ----------
def xml_response(x:str)->Response:
    return Response(x, mimetype="text/xml; charset=utf-8")

def ws_absolute_url(path:str)->str:
    host = PUBLIC_BASE.replace("https://","").replace("http://","")
    scheme = "wss" if PUBLIC_BASE.startswith("https://") else "ws"
    return f"{scheme}://{host}{path}"

def build_reconnect_twiml(say_text: str) -> str:
    vr = VoiceResponse()
    if say_text.strip():
        vr.say(say_text, language="ja-JP")
    with vr.connect() as c:
        c.stream(
            url=ws_absolute_url("/media"),
            track="inbound_track",
            status_callback=f"{PUBLIC_BASE}/ms-status",
            status_callback_method="POST",
            status_callback_event="start end"
        )
    return str(vr)


# ---------- health/version ----------
@app.route("/healthz", methods=["GET"])
def healthz(): return Response("ok", mimetype="text/plain")

@app.route("/version", methods=["GET"])
def version():
    return Response(json.dumps({
        "app": APP_NAME,
        "ws_url": ws_absolute_url("/media"),
        "public_base": PUBLIC_BASE
    }), mimetype="application/json")


# ---------- 初回 TwiML（挨拶→Stream） ----------
def build_initial_twiml() -> str:
    vr = VoiceResponse()
    vr.say("こんにちは。こちらは受付です。", language="ja-JP")
    vr.pause(length=1)
    vr.say("このあとご用件をお話しください。", language="ja-JP")
    with vr.connect() as c:
        c.stream(
            url=ws_absolute_url("/media"),
            track="inbound_track",
            status_callback=f"{PUBLIC_BASE}/ms-status",
            status_callback_method="POST",
            status_callback_event="start end"
        )
    log.info(f"[TwiML] stream url -> {ws_absolute_url('/media')}")
    return str(vr)

@app.route("/voice", methods=["GET","POST"])
def voice(): return xml_response(build_initial_twiml())


# ---------- Media Streams status callback ----------
@app.route("/ms-status", methods=["POST"])
def ms_status():
    payload = request.form or request.json or {}
    event = payload.get("StreamEvent") or payload.get("StatusCallbackEvent") or "unknown"
    call_sid = payload.get("CallSid") or payload.get("callSid")
    stream_sid = payload.get("StreamSid") or payload.get("streamSid")
    reason = payload.get("Reason") or payload.get("StopReason")
    log.info(f"[MS-STATUS] event={event} callSid={call_sid} streamSid={stream_sid} reason={reason} raw={dict(payload)}")
    return Response("ok", mimetype="text/plain")


# ---------- μ-law -> PCM16 / WAV ----------
def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    if not ulaw: return b""
    if not hasattr(ulaw_to_pcm16,"_t"):
        t=[]
        for i in range(256):
            u=~i & 0xFF; sign=u&0x80; exp=(u>>4)&7; man=u&0x0F
            mag=((man<<3)+0x84)<<exp; s=mag-0x84; s=-s if sign else s
            s=max(-32768,min(32767,s)); t.append(s & 0xFFFF)
        ulaw_to_pcm16._t=t
    out=bytearray()
    for b in ulaw: s=ulaw_to_pcm16._t[b]; out+=bytes((s&0xFF,(s>>8)&0xFF))
    return bytes(out)

def write_wav_8k_pcm16(pcm16: bytes, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16)


# ---------- VAD（内容ベース） ----------
class ContentVAD:
    """
    μ-law 1バイト=1サンプル。0xFF が “デジタルサイレンス” とみなされることが多い。
    1フレーム内で 0xFF が SILENCE_RATIO 以上なら無音。そうでなければ“有声”。
    """
    def __init__(self, idle_ms:int):
        self.idle_ms = idle_ms
        self.last_voiced_ts = time.time()  # 最後に“有声”を検出した時刻
    def on_frame(self, ulaw: bytes, silence_ratio: float) -> None:
        if not ulaw:
            return
        silent_count = ulaw.count(0xFF)
        ratio = silent_count / float(len(ulaw))
        if ratio < silence_ratio:
            # 有声フレーム
            self.last_voiced_ts = time.time()
    def silence_ms(self) -> float:
        return (time.time() - self.last_voiced_ts) * 1000.0
    def is_segment_boundary(self) -> bool:
        return self.silence_ms() > self.idle_ms


# ---------- バッファ ----------
class CallBuffer:
    def __init__(self, call_sid:str):
        self.call_sid=call_sid
        self.ulaw_chunks: List[bytes]=[]
    def append(self, b:bytes):
        if b: self.ulaw_chunks.append(b)
    def total_ms(self)->int:
        return int(sum(len(c) for c in self.ulaw_chunks) / 8)  # ≒ 1,000ms/8,000B
    def reset(self): self.ulaw_chunks.clear()
    def export_wav(self)->Optional[str]:
        if not self.ulaw_chunks: return None
        pcm16 = ulaw_to_pcm16(b"".join(self.ulaw_chunks))
        path = f"/tmp/{self.call_sid}_{uuid.uuid4().hex}.wav"
        write_wav_8k_pcm16(pcm16, path)
        return path


# ---------- Whisper → GPT → TwiML更新 ----------
def run_pipeline_and_reply(call_sid:str, wav_path:Optional[str]):
    text=""
    if wav_path and oai_client:
        try:
            with open(wav_path,"rb") as f:
                tr = oai_client.audio.transcriptions.create(model="whisper-1", file=f)
            text = (tr.text or "").strip()
        except Exception as e:
            log.warning(f"[ASR] Whisper error: {e}")
    log.info(f"[ASR] text='{text}'")

    reply="恐れ入ります、もう一度ゆっくりお話しください。"
    if text and oai_client:
        try:
            cr = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"あなたは丁寧な日本語の電話受付AIです。簡潔に1〜2文で答えてください。"},
                    {"role":"user","content":text},
                ],
                temperature=0.4,
            )
            reply = cr.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"[NLG] Chat error: {e}")
    log.info(f"[NLG] reply='{reply}'")

    twiml = build_reconnect_twiml(reply)
    twilio_client.calls(call_sid).update(twiml=twiml)
    log.info(f"[TwiML-UPDATE] replied and reconnected stream for callSid={call_sid}")

    try:
        if wav_path and os.path.exists(wav_path): os.remove(wav_path)
    except Exception: pass


# ---------- 事前ログ ----------
@app.before_request
def _pre_log():
    if request.path == "/media":
        log.info(f"[PRE] /media method={request.method} upgrade={request.headers.get('Upgrade')} "
                 f"origin={request.headers.get('Origin')} ua={request.headers.get('User-Agent')}")


# ---------- WebSocket: /media ----------
from collections import defaultdict
states: Dict[str, Dict[str,int]] = defaultdict(lambda: {"frames":0})
buffers: Dict[str, CallBuffer] = {}
vads: Dict[str, ContentVAD] = {}

@sock.route("/media")
def media_ws(ws):
    call_sid=None
    try:
        log.info("[WS] connected (handshake OK)")
        while True:
            raw = ws.receive()
            if raw is None:
                log.info("[WS] client closed"); break
            try:
                data = json.loads(raw)
            except Exception:
                continue

            ev = data.get("event")
            if ev == "start":
                call_sid = data.get("start",{}).get("callSid")
                stream_sid = data.get("start",{}).get("streamSid")
                buffers[call_sid] = CallBuffer(call_sid)
                vads[call_sid] = ContentVAD(VAD_IDLE_MS)
                states[call_sid]["frames"]=0
                log.info(f"[STATUS] stream-started callSid={call_sid} streamSid={stream_sid}")

            elif ev == "media" and call_sid:
                payload = data.get("media",{}).get("payload","")
                if not payload: continue

                ulaw = base64.b64decode(payload)
                buffers[call_sid].append(ulaw)
                states[call_sid]["frames"] += 1

                # 有声検出：フレーム内容から無音/有声を判定
                vads[call_sid].on_frame(ulaw, SILENCE_RATIO)

                # 区切り判定：十分溜まっていて無音が続いたら切り出し
                if buffers[call_sid].total_ms() >= MIN_UTTER_MS and vads[call_sid].is_segment_boundary():
                    wav_path = buffers[call_sid].export_wav()
                    buffers[call_sid].reset()
                    threading.Thread(
                        target=run_pipeline_and_reply, args=(call_sid, wav_path), daemon=True
                    ).start()

                if states[call_sid]["frames"] % 50 == 0:
                    log.info(f"[MEDIA] callSid={call_sid} frames={states[call_sid]['frames']}")

            elif ev == "stop":
                log.info(f"[STATUS] stream-stopped callSid={call_sid}")
                break

    except Exception as e:
        log.warning(f"[WS] exception: {e}")
    finally:
        buffers.pop(call_sid, None); vads.pop(call_sid, None); states.pop(call_sid, None)
        log.info(f"[WS] closed callSid={call_sid}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")), debug=False)
