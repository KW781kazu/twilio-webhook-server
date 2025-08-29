# =========================
# main_inbound_turn.py
# Twilio(Media Streams) -> Whisper(ja固定) -> GPT(履歴あり) -> <Say> -> reconnect <Stream>
# 改良点：
#  - VADにデバウンス/連打防止（境界通過のみ発火・最小長チェック・連続区切り間隔）
#  - Whisper language="ja"
#  - 会話履歴を call_sid ごとに保持
#  - TwiML更新は逐次＆stop後はスキップ
# =========================
import os, sys, json, time, base64, wave, uuid, logging, threading, struct
from typing import Optional, Dict, List
from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

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

# ===== VAD parameters (env で調整可) =====
VAD_IDLE_MS     = int(os.environ.get("VAD_IDLE_MS", "1100"))  # 有声→無声の連続がこの時間を超えたら区切り
MIN_UTTER_MS    = int(os.environ.get("MIN_UTTER_MS", "900"))  # 最小発話長
BOUND_GAP_MS    = int(os.environ.get("BOUND_GAP_MS", "1200")) # 区切り後しばらく再発火禁止
RMS_THRESH      = float(os.environ.get("RMS_THRESH", "230.0"))
PEAK_THRESH     = int(os.environ.get("PEAK_THRESH", "850"))

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
        "public_base": PUBLIC_BASE,
        "vad": {
            "idle_ms": VAD_IDLE_MS, "min_utter_ms": MIN_UTTER_MS,
            "bound_gap_ms": BOUND_GAP_MS, "rms": RMS_THRESH, "peak": PEAK_THRESH
        }
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
    if call_sid and event in ("stream-stopped", "end", "stop"):
        st = CALLS.get(call_sid)
        if st:
            st["closed"] = True
            with st["cond"]:
                st["cond"].notify_all()
    return Response("ok", mimetype="text/plain")


# ---------- μ-law <-> PCM16 / WAV ----------
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

def pcm16_rms_and_peak(pcm16: bytes) -> (float, int):
    if not pcm16: return 0.0, 0
    cnt = len(pcm16)//2
    if cnt == 0: return 0.0, 0
    fmt = "<%dh" % cnt
    samples = struct.unpack(fmt, pcm16[:cnt*2])
    peak = max(abs(s) for s in samples)
    acc = 0
    for s in samples: acc += s*s
    rms = (acc / cnt) ** 0.5
    return rms, peak

def write_wav_8k_pcm16(pcm16: bytes, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16)


# ---------- VAD（デバウンス） ----------
class DebouncedVAD:
    """
    フレームごとにRMS/Peakで有声判定。
    - 直近の「有声」からの無音継続時間 > VAD_IDLE_MS になった瞬間だけ True を返す（境界通過）
    - 直近の区切りから BOUND_GAP_MS 経過するまで再発火しない
    """
    def __init__(self, idle_ms:int, gap_ms:int):
        self.idle_ms = idle_ms
        self.gap_ms  = gap_ms
        self.last_voiced_ts = time.time()
        self.last_boundary_ts = 0.0

    def on_frame_pcm16(self, pcm16: bytes):
        rms, peak = pcm16_rms_and_peak(pcm16)
        voiced = (rms >= RMS_THRESH) or (peak >= PEAK_THRESH)
        now = time.time()
        if voiced:
            self.last_voiced_ts = now
        silence_ms = (now - self.last_voiced_ts) * 1000.0
        boundary = False
        if silence_ms > self.idle_ms and (now - self.last_boundary_ts) * 1000.0 > self.gap_ms:
            boundary = True
            self.last_boundary_ts = now
        return rms, peak, voiced, silence_ms, boundary


# ---------- per-call state ----------
CALLS: Dict[str, Dict] = {}  # call_sid -> {buffer, vad, frames, processing, closed, messages, lock, cond}

class CallBuffer:
    def __init__(self, call_sid:str):
        self.call_sid=call_sid
        self.ulaw_chunks: List[bytes]=[]
    def append(self, b:bytes):
        if b: self.ulaw_chunks.append(b)
    def total_ms(self)->int:
        return int(sum(len(c) for c in self.ulaw_chunks) / 8)
    def reset(self): self.ulaw_chunks.clear()
    def export_wav(self)->Optional[str]:
        if not self.ulaw_chunks: return None
        pcm16 = ulaw_to_pcm16(b"".join(self.ulaw_chunks))
        path = f"/tmp/{self.call_sid}_{uuid.uuid4().hex}.wav"
        write_wav_8k_pcm16(pcm16, path)
        return path


# ---------- Whisper → GPT → TwiML更新 ----------
def run_pipeline_and_reply(call_sid:str, wav_path:Optional[str]):
    st = CALLS.get(call_sid)
    if not st: return

    with st["lock"]:
        if st["processing"] or st["closed"]:
            return
        st["processing"] = True

    try:
        text=""
        if wav_path and oai_client:
            try:
                with open(wav_path,"rb") as f:
                    tr = oai_client.audio.transcriptions.create(
                        model="whisper-1", file=f, language="ja"
                    )
                text = (tr.text or "").strip()
            except Exception as e:
                log.warning(f"[ASR] Whisper error: {e}")
        log.info(f"[ASR] text='{text}'")

        # --- GPT with conversation memory ---
        reply="恐れ入ります、もう一度ゆっくりお話しください。"
        st_msgs = st["messages"]
        if text:
            st_msgs.append({"role":"user","content":text})
        if oai_client and st_msgs:
            try:
                cr = oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st_msgs,
                    temperature=0.4,
                )
                reply = cr.choices[0].message.content.strip()
                st_msgs.append({"role":"assistant","content":reply})
            except Exception as e:
                log.warning(f"[NLG] Chat error: {e}")
        log.info(f"[NLG] reply='{reply}'")

        if st["closed"]:
            log.info(f"[TwiML-UPDATE] skipped (call closed) callSid={call_sid}")
            return

        twiml = build_reconnect_twiml(reply)
        try:
            twilio_client.calls(call_sid).update(twiml=twiml)
            log.info(f"[TwiML-UPDATE] replied and reconnected stream for callSid={call_sid}")
        except TwilioRestException as e:
            log.warning(f"[Twilio UPDATE] failed: {e.msg} (status={e.status})")

    finally:
        try:
            if wav_path and os.path.exists(wav_path): os.remove(wav_path)
        except Exception:
            pass
        with st["lock"]:
            st["processing"] = False
        with st["cond"]:
            st["cond"].notify_all()


# ---------- 事前ログ ----------
@app.before_request
def _pre_log():
    if request.path == "/media":
        log.info(f"[PRE] /media method={request.method} upgrade={request.headers.get('Upgrade')} "
                 f"origin={request.headers.get('Origin')} ua={request.headers.get('User-Agent')}")


# ---------- WebSocket: /media ----------
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
                CALLS[call_sid] = {
                    "buffer": CallBuffer(call_sid),
                    "vad":    DebouncedVAD(VAD_IDLE_MS, BOUND_GAP_MS),
                    "frames": 0,
                    "processing": False,
                    "closed": False,
                    "messages": [
                        {"role":"system","content":"あなたは丁寧で簡潔な日本語の電話受付AIです。相手の意図を確認し、必要なら追加の短い質問を1つ返してください。"},
                    ],
                    "lock": threading.Lock(),
                    "cond": threading.Condition(),
                }
                log.info(f"[STATUS] stream-started callSid={call_sid} streamSid={stream_sid}")

            elif ev == "media" and call_sid:
                st = CALLS.get(call_sid); 
                if not st: continue
                payload = data.get("media",{}).get("payload","")
                if not payload: continue

                ulaw = base64.b64decode(payload)
                pcm16 = ulaw_to_pcm16(ulaw)

                rms, peak, voiced, silence_ms, boundary = st["vad"].on_frame_pcm16(pcm16)

                st["buffer"].append(ulaw)
                st["frames"] += 1

                # 境界通過かつ 最小長 条件で切り出し（逐次・連打防止）
                if boundary and st["buffer"].total_ms() >= MIN_UTTER_MS and not st["processing"]:
                    wav_path = st["buffer"].export_wav()
                    st["buffer"].reset()
                    threading.Thread(target=run_pipeline_and_reply, args=(call_sid, wav_path), daemon=True).start()

                if st["frames"] % 50 == 0:
                    log.info(f"[MEDIA] callSid={call_sid} frames={st['frames']} rms={rms:.0f} peak={peak} silence_ms={silence_ms:.0f} voiced={voiced}")

            elif ev == "stop":
                st = CALLS.get(call_sid)
                if st:
                    st["closed"] = True
                    # 最後の残りを1回だけ（任意）
                    if st["buffer"].total_ms() >= 500 and not st["processing"]:
                        wav_path = st["buffer"].export_wav()
                        st["buffer"].reset()
                        threading.Thread(target=run_pipeline_and_reply, args=(call_sid, wav_path), daemon=True).start()
                log.info(f"[STATUS] stream-stopped callSid={call_sid}")
                break

    except Exception as e:
        log.warning(f"[WS] exception: {e}")
    finally:
        if call_sid:
            CALLS.pop(call_sid, None)
        log.info(f"[WS] closed callSid={call_sid}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","10000")), debug=False)
