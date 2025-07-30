import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import speech
from vertexai.generative_models import GenerativeModel
import vertexai

# --- 環境変数から読み込み ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"  # Gemini対応リージョン
MODEL_ID = "gemini-2.5-flash-lite"

# Vertex AI 初期化
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_ID)

# Flask アプリ
app = Flask(__name__)

# Twilio: 電話着信時の応答
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    resp.say("こんにちは。お名前とご用件をお話しください。")
    resp.record(
        action="/recording",
        max_length=30,
        finish_on_key="#",
        play_beep=True
    )
    return Response(str(resp), mimetype="text/xml")

# Twilio: 録音処理
@app.route("/recording", methods=["POST"])
def recording():
    recording_url = request.form.get("RecordingUrl")
    print(f"[INFO] Received recording: {recording_url}")

    # 音声をテキストに変換
    transcription = transcribe_audio(recording_url)
    print(f"[INFO] Transcription: {transcription}")

    # Geminiで応答生成
    ai_response = generate_gemini_response(transcription)
    print(f"[INFO] Gemini Response: {ai_response}")

    # 応答を返す
    resp = VoiceResponse()
    resp.say(ai_response)
    return Response(str(resp), mimetype="text/xml")

# Google Speech-to-Text で文字起こし
def transcribe_audio(audio_url):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=audio_url + ".wav")
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="ja-JP"
    )
    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

# Gemini APIで応答生成
def generate_gemini_response(user_input):
    prompt = f"以下の内容に丁寧に返答してください：{user_input}"
    response = model.generate_content(prompt)
    return response.text if response else "すみません、今は応答できません。"

# Render実行
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
