
# main_inbound_turn.py
# 片方向 Media Streams: 呼→Twilio→サーバ（ASR/LLM）→Twilio <Say> 再生→即Stream再開
import os, json, time, base64, audioop, wave, io, threading
from queue import Queue
from datetime import datetime, timezone

from flask import Flask, request, Response
from flask_sock import Sock
from dotenv import load_dotenv
import requests
from twilio.rest import Client

load_dotenv()

# === 環境変数 ===
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_NUMBER      = os.getenv("TWILIO_NUMBER", "")
PUBLIC_BASE        = os.getenv("PUBLIC_BASE", "").rstrip("/")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_ASR   = os.getenv("OPENAI_MODEL", "gpt-4o-mini-transcribe")
OPENAI_MODEL_CHAT  = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TRACK_MODE         = "inbound"

assert PUBLIC_BASE.startswith("https://"), "PUBLIC_BASE は https で始まる必要があります"

# === Flask ===
app = Flask(__name__)
sock = Sock(app)

tw_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

INSTRUCTIONS_JA = (
    "あなたは「フロントガラス修理ナビ」のAI電話受付アシスタントです。"
    "丁寧だが長すぎない口語で、相手の直近の発話にまず短く共感し、つづけて次の1点だけ質問します。"
    "収集すべき項目: 破損箇所/見積希望/車検証の有無/郵便番号/車種/初年度登録/車台番号/型式指定(5桁)/類別区分(4桁)。"
    "雑談は短く、やさしく本筋に戻る。"
)

# 録音やログの保存先
os.makedirs("recordings", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def twiml_stream_url():
    return PUBLIC_BASE + "/media"

def twiml_status_url():
    return PUBLIC_BASE + "/stream-status"

# ============== 1. 着信時のTwiML ==============
@app.route("/voice", methods=["POST"])
def voice():
    # ここで <Connect><Stream> を張り、statusCallback を POST に
    twiml = f"""
<Response>
  <Say language="ja-JP" voice="Polly.Mizuki">フロントガラス修理ナビです。AIアシスタントが受付いたします。どうぞよろしくお願いいたします。</Say>
  <Connect>
    <Stream track="{TRACK_MODE}_track"
            url="{twiml_stream_url()}"
            statusCallbackMethod="POST"
            statusCallback="{twiml_status_url()}" />
  </Connect>
</Response>
""".strip()
    return Response(twiml, mimetype="text/xml")

# ============== 2. Stream ステータス受け取り（任意でログ） ==============
@app.route("/stream-status", methods=["POST"])
def stream_status():
    # Twilio からの状態通知
    with open("logs/stream_status.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "form": request.form.to_dict()
        }, ensure_ascii=False) + "\n")
    return ("", 204)

# ============== 3. 片方向メディアWS ==============
ULAW_FRAME = 160  # 20ms @8k

class TurnBuffer:
    """シンプルVAD: 連続無音でターンを確定"""
    def __init__(self):
        self.buf = bytearray()
        self.speaking = False
        self.sil_count = 0
        self.on_count  = 0
        self.thr_on = 1100
        self.thr_off = 800

    def feed_ulaw(self, ulaw_bytes):
        pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)
        rms = audioop.rms(pcm16, 2)

        # ノイズ学習は簡略（最初200msは平均化なども可）
        if not self.speaking:
            if rms >= self.thr_on:
                self.on_count += 1
                if self.on_count >= 6:  # 120ms
                    self.speaking = True
                    self.buf.extend(pcm16)
                    self.on_count = 0
            else:
                self.on_count = 0
        else:
            self.buf.extend(pcm16)
            if rms < self.thr_off:
                self.sil_count += 1
            else:
                self.sil_count = 0

            if self.sil_count >= 12 or len(self.buf) > 16000*3*2:  # 240ms or ~3s
                data = bytes(self.buf)
                self.buf.clear()
                self.speaking = False
                self.sil_count = 0
                return data
        return None

def pcm16_to_wav_bytes(pcm16_8k: bytes) -> bytes:
    # Whisper系は16k推奨だが mini-transcribe は8kでもOK。16kへ補間して渡す
    pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16_16k)
    return bio.getvalue()

def transcribe_wav(wav_bytes: bytes) -> str:
    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav"),
    }
    data = {"model": OPENAI_MODEL_ASR, "language": "ja"}
    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        data=data, files=files, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["text"].strip()

def chat_reply(text: str) -> str:
    sys = INSTRUCTIONS_JA
    usr = f"お客様の直近の発話:「{text}」。簡潔に共感→次の1問だけを聞いて。"
    payload = {
        "model": OPENAI_MODEL_CHAT,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ],
        "max_tokens": 120
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                               "Content-Type": "application/json"},
                      data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def update_call_twiml(call_sid: str, say_text: str):
    # 返事を再生→すぐ Stream を張り直す
    twiml = f"""
<Response>
  <Say language="ja-JP" voice="Polly.Mizuki">{say_text}</Say>
  <Connect>
    <Stream track="{TRACK_MODE}_track"
            url="{twiml_stream_url()}"
            statusCallbackMethod="POST"
            statusCallback="{twiml_status_url()}" />
  </Connect>
</Response>
""".strip()
    tw_client.calls(call_sid).update(twiml=twiml)

@sock.route("/media")
def media(ws):
    # 1通話ごとに状態を持つ
    call_sid = None
    turn = TurnBuffer()

    # 保存用
    rec_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcm_path = f"recordings/{rec_id}.pcm8k"
    txt_path = f"recordings/{rec_id}.txt"
    fpcm = open(pcm_path, "ab")

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            evt = json.loads(msg)
            typ = evt.get("event")

            if typ == "start":
                call_sid = evt["start"]["callSid"]
                continue

            if typ == "media":
                b64 = evt["media"]["payload"]
                ulaw = base64.b64decode(b64)
                # 保存（デバッグ用）
                fpcm.write(audioop.ulaw2lin(ulaw, 2))

                # ターン確定?
                seg = turn.feed_ulaw(ulaw)
                if seg:
                    # 1ターン処理（別スレッドで即時化）
                    threading.Thread(
                        target=process_turn,
                        args=(call_sid, seg, txt_path),
                        daemon=True
                    ).start()

            if typ == "stop":
                break
    finally:
        try: fpcm.close()
        except: pass

def process_turn(call_sid: str, pcm16_8k: bytes, txt_path: str):
    try:
        wav = pcm16_to_wav_bytes(pcm16_8k)
        user_text = transcribe_wav(wav)
        if not user_text:
            return
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"[U] {user_text}\n")

        reply = chat_reply(user_text)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"[A] {reply}\n")

        update_call_twiml(call_sid, reply)
    except Exception as e:
        with open("logs/errors.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} process_turn ERR: {repr(e)}\n")

# ============== エントリ ==============
if __name__ == "__main__":
    print(f">>> Starting server on http://0.0.0.0:5000 (TRACK_MODE: {TRACK_MODE})")
    app.run(host="0.0.0.0", port=5000, debug=True)
