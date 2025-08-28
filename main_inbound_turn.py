# main_inbound_turn.py  — Twilio Media Streams (inbound) × Whisper × gpt-4o-mini
# 目的: 出だし欠け/取りこぼし/間の長さを抑えつつ自然に返答

import os, json, time, base64, audioop, wave, io, threading
from datetime import datetime, timezone
from flask import Flask, request, Response
from flask_sock import Sock
from dotenv import load_dotenv
import requests
from twilio.rest import Client

load_dotenv()

# ========== 環境変数 ==========
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_ASR   = os.getenv("OPENAI_MODEL_ASR", "whisper-1")
OPENAI_MODEL_CHAT  = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
PUBLIC_BASE        = os.getenv("PUBLIC_BASE", "").rstrip("/")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TRACK_MODE         = "inbound"   # Twilio 側は inbound_track のみ

assert PUBLIC_BASE.startswith("https://"), "PUBLIC_BASE は https:// で始めてください"

# ========== Flask / Twilio ==========
app = Flask(__name__)
sock = Sock(app)
tw = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

INSTRUCTIONS_JA = (
    "あなたは「フロントガラス修理ナビ」のAI電話受付アシスタントです。"
    "丁寧だが長すぎない口語で、相手の直近の発話へ短く共感→“次の1問だけ”を聞く。"
    "収集: 破損箇所/見積希望/車検証の有無/郵便番号/車種/初年度登録/車台番号/型式指定(5桁)/類別区分(4桁)。"
    "雑談は短く、やさしく本筋へ戻す。"
)

os.makedirs("recordings", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def stream_ws_url():   return PUBLIC_BASE.replace("https://", "wss://") + "/media"
def status_cb_url():   return PUBLIC_BASE + "/stream-status"

# ヘルスチェック用（Render で 200 を返すエンドポイント）
@app.get("/")
def root_ok():
    return "ok", 200

# ========== TwiML（着信時） ==========
@app.route("/voice", methods=["POST"])
def voice():
    twiml = f"""
<Response>
  <Say language="ja-JP" voice="Polly.Mizuki">フロントガラス修理ナビです。AIアシスタントが受付いたします。どうぞよろしくお願いいたします。</Say>
  <Connect>
    <Stream track="{TRACK_MODE}_track"
            url="{stream_ws_url()}"
            statusCallbackMethod="POST"
            statusCallback="{status_cb_url()}" />
  </Connect>
</Response>
""".strip()
    return Response(twiml, mimetype="text/xml")

# ========== Stream 状態ログ ==========
@app.route("/stream-status", methods=["POST"])
def stream_status():
    with open("logs/stream_status.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            "form": request.form.to_dict()}, ensure_ascii=False) + "\n")
    return ("", 204)

# ========== VAD（適応 + 前後パディング + 最短長） ==========
# 20ms@8k μ-law フレーム
ULAW_FRAME = 160

class AdaptiveVAD:
    """最初の1秒でノイズ平均→しきい値決定。語頭160ms/語尾280ms基準、最短600ms、上限3.5s。"""
    def __init__(self):
        self.learn_frames = 50       # 1s (50*20ms)
        self.learn_sum = 0
        self.learn_cnt = 0
        self.thr_on  = 1200
        self.thr_off = 850
        self.in_speech = False
        self.on_cnt = 0
        self.off_cnt = 0
        self.buf = bytearray()
        self.pre = bytearray()       # 語頭パディング用
        self.min_start_frames = 8    # 160ms
        self.min_end_frames   = 14   # 280ms
        self.min_seg_ms       = 600
        self.max_seg_ms       = 3500

    def feed(self, ulaw_bytes):
        pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)
        rms = audioop.rms(pcm16, 2)

        # 常に200msプリロールを保持（語頭欠け防止）
        self.pre += pcm16
        max_pre = int(0.2 * 8000) * 2
        if len(self.pre) > max_pre:
            self.pre = self.pre[-max_pre:]

        # 学習フェーズ
        if self.learn_cnt < self.learn_frames:
            self.learn_sum += rms
            self.learn_cnt += 1
            if self.learn_cnt == self.learn_frames:
                avg = self.learn_sum / max(1, self.learn_cnt)
                self.thr_on  = max(int(avg*2.0 + 250), 1100)
                self.thr_off = max(int(avg*1.3 + 120), 800)
                print(f"[VAD] learned avg={avg:.1f} -> thr_on={self.thr_on}, thr_off={self.thr_off}")
            return None

        # 判定
        if not self.in_speech:
            if rms >= self.thr_on:
                self.on_cnt += 1
                if self.on_cnt >= self.min_start_frames:
                    self.in_speech = True
                    self.buf += self.pre   # 語頭にプリロールを付与
                    self.buf += pcm16
                    self.on_cnt = 0
            else:
                self.on_cnt = 0
        else:
            self.buf += pcm16
            seg_ms = int(len(self.buf)/2/8000*1000)
            if rms < self.thr_off:
                self.off_cnt += 1
            else:
                self.off_cnt = 0

            if self.off_cnt >= self.min_end_frames or seg_ms >= self.max_seg_ms:
                data = bytes(self.buf)
                self.buf.clear()
                self.off_cnt = 0
                self.in_speech = False
                if seg_ms < self.min_seg_ms:
                    return None
                return data
        return None

def pcm8k_to_wav16k_norm(pcm16_8k: bytes) -> bytes:
    # 8k→16kへ補間
    pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)
    # 目標 -3 dBFS に正規化
    peak = max(1, audioop.max(pcm16_16k, 2))
    target = int(32767 * (10**(-3/20)))
    gain = min(3.0, target/peak)
    pcm16_16k = audioop.mul(pcm16_16k, 2, gain)
    # 前後100msの無音パッド
    pad = b"\x00" * int(0.1 * 16000) * 2
    pcm16_16k = pad + pcm16_16k + pad

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(pcm16_16k)
    return bio.getvalue()

def whisper_transcribe(wav_bytes: bytes, lang="ja") -> str:
    files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
    data  = {"model": OPENAI_MODEL_ASR, "language": lang}
    r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                      data=data, files=files, timeout=60)
    r.raise_for_status()
    return (r.json().get("text") or "").strip()

def chat_reply(user_text: str) -> str:
    sys = INSTRUCTIONS_JA
    usr = f"お客様の直近の発話:「{user_text}」。短く共感→次の1問だけ聞いて。"
    payload = {"model": OPENAI_MODEL_CHAT,
               "messages":[{"role":"system","content":sys},
                           {"role":"user","content":usr}],
               "max_tokens": 120, "temperature": 0.3}
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                               "Content-Type":"application/json"},
                      data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def ssml(text: str) -> str:
    esc = (text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    return f'<speak><prosody rate="97%" pitch="+2st" volume="+1dB">{esc}</prosody></speak>'

def twiml_say_again(call_sid: str, text: str):
    twiml = f"""
<Response>
  <Say language="ja-JP" voice="Polly.Mizuki">{ssml(text)}</Say>
  <Connect>
    <Stream track="{TRACK_MODE}_track"
            url="{stream_ws_url()}"
            statusCallbackMethod="POST"
            statusCallback="{status_cb_url()}" />
  </Connect>
</Response>
""".strip()
    tw.calls(call_sid).update(twiml=twiml)

# ========== Media WS（片方向） ==========
@sock.route("/media")
def media(ws):
    call_sid = None
    vad = AdaptiveVAD()

    rec_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_pcm_path = f"recordings/{rec_id}.pcm8k"
    txt_log_path = f"recordings/{rec_id}.txt"
    fpcm = open(raw_pcm_path, "ab")

    try:
        while True:
            msg = ws.receive()
            if msg is None: break
            evt = json.loads(msg); et = evt.get("event")

            if et == "start":
                call_sid = evt["start"]["callSid"]
                print(f"[WS] connected (handshake OK) callSid={call_sid}")
                continue

            if et == "media":
                ulaw = base64.b64decode(evt["media"]["payload"])
                pcm16 = audioop.ulaw2lin(ulaw, 2)
                fpcm.write(pcm16)
                seg = vad.feed(ulaw)
                if seg:
                    threading.Thread(target=_process_turn,
                                     args=(call_sid, seg, txt_log_path),
                                     daemon=True).start()
                continue

            if et == "stop":
                print("[WS] stop event")
                break
    finally:
        try: fpcm.close()
        except: pass

def _process_turn(call_sid: str, pcm16_8k: bytes, txt_path: str):
    try:
        wav = pcm8k_to_wav16k_norm(pcm16_8k)
        turn_id = datetime.now().strftime("%H%M%S_%f")
        with open(f"recordings/{turn_id}.wav", "wb") as wf:
            wf.write(wav)

        t0 = time.time()
        user_text = whisper_transcribe(wav, lang="ja")
        t_asr = int((time.time()-t0)*1000)
        if not user_text: return

        reply = chat_reply(user_text)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"[{turn_id}] U:{user_text}\n[{turn_id}] A:{reply}\n")

        print(f"[TURN] ASR {t_asr}ms  text='{user_text}'  -> reply='{reply[:32]}…'")
        twiml_say_again(call_sid, reply)

    except Exception as e:
        with open("logs/errors.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} process_turn ERR: {repr(e)}\n")

# ========== Entrypoint（ローカル用） ==========
if __name__ == "__main__":
    print(f">>> Starting server on http://0.0.0.0:5000 (TRACK_MODE={TRACK_MODE})")
    print("PUBLIC_BASE =", PUBLIC_BASE)
    print("Stream WS  =", stream_ws_url())
    app.run(host="0.0.0.0", port=5000, debug=True)
