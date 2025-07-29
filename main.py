from flask import Flask, request, jsonify
import requests
import os
import json
import io
from google.cloud import speech

app = Flask(__name__)

# --- Google認証の修正 ---
# 環境変数にJSON文字列が入っているので、起動時にファイル化する
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    creds_path = "/tmp/service-account.json"  # Renderの一時ディレクトリに書き出し
    with open(creds_path, "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
# これで speech.SpeechClient() が正常に認証できる

@app.route("/recording", methods=["POST"])
def process_recording():
    try:
        app.logger.info("=== /recording endpoint called ===")
        recording_url = request.form.get("RecordingUrl")
        app.logger.info(f"Recording URL: {recording_url}")
        if not recording_url:
            app.logger.error("No RecordingUrl provided")
            return jsonify({"error": "No RecordingUrl"}), 400

        # Twilioの録音ファイルを取得
        app.logger.info("Downloading audio from Twilio...")
        audio_response = requests.get(f"{recording_url}.wav")
        if audio_response.status_code != 200:
            app.logger.error(f"Failed to download audio: {audio_response.status_code}")
            return jsonify({"error": "Failed to download audio"}), 500

        audio_content = audio_response.content
        app.logger.info(f"Downloaded audio size: {len(audio_content)} bytes")

        # Google Speech-to-Text
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
