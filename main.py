from flask import Flask, request, Response
import requests
import os
from google.cloud import speech

app = Flask(__name__)

# Twilio認証情報
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

@app.route("/voice", methods=["POST"])
def voice():
    response = """
    <Response>
        <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
        <Record 
            action="/recording" 
            method="POST" 
            maxLength="30" 
            finishOnKey="*" 
            playBeep="true" />
    </Response>
    """
    return Response(response, mimetype="application/xml")

@app.route("/recording", methods=["POST"])
def process_recording():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            return Response("<Response><Say>録音URLが取得できませんでした。</Say></Response>", mimetype="application/xml")

        # Twilio録音ファイルを取得
        audio_response = requests.get(
            f"{recording_url}.wav",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        if audio_response.status_code != 200:
            return Response("<Response><Say>音声の取得に失敗しました。</Say></Response>", mimetype="application/xml")

        audio_content = audio_response.content

        # Google Speech-to-Text
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="ja-JP"
        )
        response = client.recognize(config=config, audio=audio)

        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            reply = f"<Response><Say language='ja-JP'>ありがとうございます。あなたはこう言いました。{transcript}</Say></Response>"
        else:
            reply = "<Response><Say language='ja-JP'>すみません。音声を認識できませんでした。</Say></Response>"

        return Response(reply, mimetype="application/xml")

    except Exception as e:
        error_reply = f"<Response><Say>サーバーエラーが発生しました: {str(e)}</Say></Response>"
        return Response(error_reply, mimetype="application/xml")

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
