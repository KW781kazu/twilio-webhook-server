from flask import Flask, request, jsonify, Response
import requests
import os
import json
from google.cloud import speech

app = Flask(__name__)

# --- Google認証の修正 ---
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    creds_path = "/tmp/service-account.json"
    with open(creds_path, "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Twilioが最初に呼ぶエンドポイント
@app.route("/voice", methods=["POST"])
def voice():
    # TwiMLを返して応答＆録音開始
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
        app.logger.info(f"Recording URL: {recording_url}")
        if not recording_url:
            app.logger.error("No RecordingUrl provided")
            return jsonify({"error": "No RecordingUrl"}), 400

        app.logger.info("Downloading audio from Twilio...")
        audio_response = requests.get(f"{recording_url}.wav")
        if audio_response.status_code != 200:
            app.logger.error(f"Failed to download audio: {audio_response.status_code}")
            return jsonify({"error": "Failed to download audio"}), 500

        audio_content = audio_response.content
        app.logger.info(f"Downloaded audio size: {len(audio_content)} bytes")

        app.logger.info("Initializing Google Speech client...")
        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="ja-JP"
        )

        app.logger.info("Sending audio to Google Speech-to-Text...")
        response = client.recognize(config=config, audio=audio)
        app.logger.info(f"Google response: {response}")

        if not response.results:
            app.logger.warning("No transcription results returned")
            return jsonify({"transcription": ""}), 200

        transcript = response.results[0].alternatives[0].transcript
        app.logger.info(f"Transcription: {transcript}")

        return jsonify({"transcription": transcript}), 200

    except Exception as e:
        app.logger.error(f"Error in /recording: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/")
def health_check():
    return "OK", 200

# --- Render用の起動 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
