from flask import Flask, request, jsonify, Response
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
        app.logger.info("=== /recording endpoint called ===")
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            return jsonify({"error": "No RecordingUrl"}), 400

        # Twilio録音ファイルを取得
        audio_response = requests.get(
            f"{recording_url}.wav",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        if audio_response.status_code != 200:
            app.logger.error(f"Failed to download audio: {audio_response.status_code}")
            return jsonify({"error": "Failed to download audio"}), 500

        audio_content = audio_response.content

        # Google Speech-to-Text
        client = speech.SpeechClient()  # ← 環境変数のパスをそのまま利用
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="ja-JP"
        )
        response = client.recognize(config=config, audio=audio)

        if not response.results:
            return jsonify({"transcription": ""}), 200

        transcript = response.results[0].alternatives[0].transcript
        return jsonify({"transcription": transcript}), 200

    except Exception as e:
        app.logger.error(f"Error in /recording: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
