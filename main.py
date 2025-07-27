from flask import Flask, request, Response
import requests
import os
import subprocess
import json
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import speech

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
GOOGLE_APPLICATION_CREDENTIALS = "/etc/secrets/credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

@app.route("/voice", methods=['POST'])
def voice():
    """最初の応答: 録音を開始"""
    resp = VoiceResponse()
    resp.say("こんにちは。ご用件をどうぞ。", language="ja-JP")
    resp.record(
        max_length=60,
        timeout=5,
        play_beep=True,
        recording_status_callback="/recording",
        recording_status_callback_event=["completed"]
    )
    return Response(str(resp), mimetype='application/xml')

@app.route("/recording", methods=['POST'])
def recording():
    """録音完了後の処理"""
    recording_url = request.form.get('RecordingUrl') + ".wav"
    print(f"TwilioからのRecordingUrl: {recording_url}")

    # 音声ファイルをダウンロード
    audio_content = requests.get(
        recording_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    ).content
    temp_input = "/tmp/input_audio.wav"
    temp_output = "/tmp/converted_audio.flac"

    with open(temp_input, "wb") as f:
        f.write(audio_content)
    print(f"音声ファイルを保存しました: {temp_input}")

    # ffmpegでFLACに変換
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ar", "16000", "-ac", "1",
            temp_output
        ], check=True)
        print(f"音声ファイルを変換しました: {temp_output}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg変換エラー: {e}")
        return Response("Error in processing audio", status=500)

    # Speech-to-Text
    try:
        print("=== Google Speech-to-Text API呼び出し開始 ===")
        client = speech.SpeechClient()
        with open(temp_output, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="ja-JP"
        )

        response = client.recognize(config=config, audio=audio)
        print(f"=== APIレスポンス ===: {response}")

        transcript = response.results[0].alternatives[0].transcript if response.results else "認識できませんでした"
        print(f"=== 認識結果 ===: {transcript}")

    except Exception as e:
        print(f"Google Speech-to-Textエラー: {e}")
        return Response("Error in Speech-to-Text", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
