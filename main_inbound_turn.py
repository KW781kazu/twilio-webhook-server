# =========================
# main_inbound_turn.py
# Twilio(Media Streams) -> Whisper(ja) -> GPT(履歴) -> <Say(Polly.Mizuki/SSML)> -> reconnect <Stream>
# 変更点:
#  - 返答再生中は Media Streams フレームを無視（ミュート）して自己反応を防止
#  - デバウンスVAD/最小長/連続区切り間隔は維持
# =========================
import os, sys, json, time, base64, wave, uuid, logging, threading, struct, re
from typing import Optional, Dict, List
from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

# OpenAI
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

# ===== VAD params =====
VAD_IDLE_MS   = int(os.environ.get("VAD_IDLE_MS", "1100"))
MIN_UTTER_MS  = int(os.environ.get("MIN_UTTER_MS", "900"))
BOUND_GAP_MS  = int(os.environ.get("BOUND_GAP_MS", "1400"))
RMS_THRESH    = float(os.environ.get("RMS_THRESH", "230.0"))
PEAK_THRESH   = int(os.environ.get("PEAK_THRESH", "850"))
MAX_UTTER_MS  = int(os.environ.get("MAX_UTTER_MS", "5000"))  # 強制切り上げ

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

def to_ssml_ja(text: str) -> str:
    t = text.strip()
    if len(t) > 140:
        t = t[:140] + "。"
    parts = [p for p in re.split(r"[。．！!？?]", t) if p]
    segs=[]
    for i,p in enumerate(parts):
        p = p.replace("、", "<break time=\"180ms\"/>")
        segs.append(f"<s>{p}</s>")
        if i != len(parts)-1:
            segs.append("<break time=\"320ms\"/>")
    body = "".join(segs) if segs else t
    return f"<speak xml:lang=\"ja-JP\">{body}</speak>"

def estimate_play_secs(text: str) -> float:
    # ざっくり日本語 7.5 文字/秒 + 小休止
    return max(1.6, len(text) / 7.5 + 0.5)

def build_reconnect_twiml(say_text: str) -> str:
    vr = VoiceResponse()
    ssml = to_ssml_ja(say_text)
    vr.say(ssml, voice="Polly.Mizuki", language="ja-JP")
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
            "bound_gap_ms": BOUND_GAP_MS, "rms": RMS_THRESH,
            "peak": PEAK_THRESH, "max_utter_ms": MAX_UTTER_MS
        }
    }), mimetype="application/json")


# ---------- 初回 TwiML ----------
def build_initial_twiml() -> str:
    vr = VoiceResponse()
    vr.say(to_ssml_ja("こんにちは。こちらは受付です。"), voice="Polly.Mizuki", language="ja-JP")
    vr.pause(length=1)
    vr.say(to_ssml_ja("このあとご用件をお話しください。"), voice="Polly.Mizuki", language="ja-JP")
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


# ---------- μ-law <-> PCM16 ----------
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
CALLS: Dict[str, Dict] = {}  # call_sid -> {...}

class CallBuffer:
    def __init__(self, call_sid:str):
        self.call_sid=call_sid
        self.ulaw_chunks: List[bytes]=[]
        self.first_ts = None
    def append(self, b:bytes):
        if b:
            if self.first_ts is None:
                self.first_ts = time.time()
            self.ulaw_chunks.append(b)
    def total_ms(self)->int:
        return int(sum(len(c) for c in self.ulaw_chunks) / 8)
    def elapsed_ms(self)->int:
        return int((time.time() - (self.first_ts or time.time())) * 1000)
    def reset(self):
        self.ulaw_chunks.clear()
        self.first_ts = None
    def export_wav(self)->Optional[str]:
        if not self.ulaw_chunks: return None
        pcm16 = ulaw_to_pcm16(b"".join(self.ulaw_chunks))
        path = f"/tmp/{self.call_sid}_{uuid.uuid4().hex}.wav"
        write_wav_8k_pcm16(pcm16, path)
        return path


# ---------- Whisper -> GPT -> TwiML ----------
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

        reply="恐れ入ります、もう一度ゆっくりお話しください。"
        msgs = st["messages"]
        if text:
            msgs.append({"role":"user","content":text})
        if oai_client and msgs:
            try:
                system_tone = (
                    "あなたは自然で丁寧な日本語の電話受付AIです。返答は1〜2文。"
                    "相手の要望を要約し、必要なら短い質問を1つだけ。オウム返しは最小限。"
                )
                if msgs and msgs[0]["role"] == "system":
                    msgs[0]["content"] = system_tone
                else:
                    msgs.insert(0, {"role":"system","content":system_tone})
                cr = oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=msgs,
                    temperature=0.3,
                    max_tokens=120,
                )
                reply = cr.choices[0].message.content.strip()
                msgs.append({"role":"assistant","content":reply})
            except Exception as e:
                log.warning(f"[NLG] Chat error: {e}")
        log.info(f"[NLG] reply='{reply}'")

        if st["closed"]:
            log.info(f"[TwiML-UPDATE] skipped (call closed) callSid={call_sid}")
            return

        twiml = build_reconnect_twiml(reply)
        try:
            twilio_client.calls(call_sid).update(twiml=twiml)
            # === 再生中は Media を無視するミュート窓を設定 ===
            mute_secs = estimate_play_secs(reply)
            st["mute_until_ts"] = time.time() + mute_secs
            log.info(f"[TwiML-UPDATE] replied & reconnected (mute {mute_secs:.1f}s) callSid={call_sid}")
        except TwilioRestException as e:
            log.warning(f"[Twilio UPDATE] failed: {e.msg} (status={e.status})")

    finally:
        try:
            if wav_path and os.path.exists(wav_path): os.remove(wav_path)
        except Exception: pass
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
                    "messages": [],
                    "lock": threading.Lock(),
                    "cond": threading.Condition(),
                    "mute_until_ts": 0.0,
                }
                log.info(f"[STATUS] stream-started callSid={call_sid} streamSid={stream_sid}")

            elif ev == "media" and call_sid:
                st = CALLS.get(call_sid)
                if not st: continue

                # === 再生中ミュート: mute_until を超えるまで完全スキップ ===
                if time.time() < float(st.get("mute_until_ts", 0.0)):
                    # VADをリセット気味にして区切りが出ないよう更新
                    st["vad"].last_voiced_ts = time.time()
                    continue

                payload = data.get("media",{}).get("payload","")
                if not payload: continue
                ulaw = base64.b64decode(payload)
                pcm16 = ulaw_to_pcm16(ulaw)

                rms, peak, voiced, silence_ms, boundary = st["vad"].on_frame_pcm16(pcm16)

                st["buffer"].append(ulaw)
                st["frames"] += 1

                long_enough = st["buffer"].elapsed_ms() >= MAX_UTTER_MS

                if (boundary or long_enough) and st["buffer"].total_ms() >= MIN_UTTER_MS and not st["processing"]:
                    wav_path = st["buffer"].export_wav()
                    st["buffer"].reset()
                    threading.Thread(target=run_pipeline_and_reply, args=(call_sid, wav_path), daemon=True).start()

                if st["frames"] % 50 == 0:
                    log.info(f"[MEDIA] callSid={call_sid} frames={st['frames']} rms={rms:.0f} peak={peak} silence_ms={silence_ms:.0f} voiced={voiced} elapsed={st['buffer'].elapsed_ms()}ms")

            elif ev == "stop":
                st = CALLS.get(call_sid)
                if st:
                    st["closed"] = True
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
