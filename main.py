from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import os
from google.cloud import speech
import vertexai
from vertexai.generative_models import GenerativeModel

app = Flask(__name__)

# Google Speech-to-Text クライアント
speech_client = speech.SpeechClient()

# Vertex AI (Gemini) 初期化
project_id = os.getenv("GCP_PROJECT_ID")
vertexai.init(project=project_id, location="us-central1")
gemini_model = GenerativeModel("gemini-1.5-flash")

@app.route("/webhook", methods=["POST"])
def webhook():
    # Twilioから送られるイベントタイプを確認
    recording_url = request.form.get("RecordingUrl")
    transcribed_text = ""

    if recording_url:
        # 録音データがある場合は取得してテキスト化
        audio_content = fetch_audio(recording_url)
        transcribed_text = transcribe_audio(audio_content)

        if not transcribed_text:
            transcribed_text = "音声を認識できませんでした。"

        # Geminiで返答生成
        response_text = generate_ai_response(transcribed_text)
    else:
        # 初回応答（録音前）
        response_text = "こんにちは。AI受付です。ご用件をお話しください。"

    # Twilioに返答
    vr = VoiceResponse()
    if not recording_url:
        # 録音する設定
        vr.say(response_text, language="ja-JP", voice="Polly.Mizuki")
        vr.record(
            action="/webhook",
            method="POST",
            max_length=10,
            play_beep=True,
            timeout=5,
        )
    else:
        # 返答だけ
        vr.say(response_text, language="ja-JP", voice="Polly.Mizuki")

    return Response(str(vr), mimetype="application/xml")

def fetch_audio(url):
    """Twilioの録音データを取得"""
    import requests
    auth = (os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    r = requests.get(f"{url}.wav", auth=auth)
    return r.content

def transcribe_audio(audio_content):
    """Google Speech-to-Textで文字起こし"""
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="ja-JP",
    )
    response = speech_client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

def generate_ai_response(text):
    """Geminiで自然な返答を生成"""
    response = gemini_model.generate_content(f"以下の内容に自然な日本語で返答してください：{text}")
    return response.text.strip()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
