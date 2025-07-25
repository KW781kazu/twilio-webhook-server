from flask import Flask, request, Response
import requests
import os
import subprocess
import time

app = Flask(__name__)

TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")

@app.route("/voice", methods=["POST"])
def voice():
    # TwiML 応答
    response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
    <Record 
        maxLength="60" 
        timeout="10" 
        finishOnKey="#" 
        recordingFormat="wav"
        recordingStatusCallback="/recording" />
</Response>"""
    return Response(response, mimetype="application/xml")

@app.route("/recording", methods=["POST"])
def recording():
    recording_url = request.form.get("RecordingUrl")
    print(f"TwilioからのRecordingUrl: {recording_url}")

    if not recording_url:
        print("録音URLが取得できませんでした")
        return "No recording URL", 400

    # 1秒待機して録音ファイルが完成するのを待つ
    time.sleep(1)

    # 録音ファイルを取得
    audio_response = requests.get(f"{recording_url}.wav", auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    if audio_response.status_code != 200:
        print(f"録音ファイル取得失敗: {audio_response.status_code}")
        return "Failed to fetch audio", 500

    # 一時ファイルに保存
    input_path = "/tmp/input_audio.wav"
    output_path = "/tmp/converted_audio.mp3"
    with open(input_path, "wb") as f:
        f.write(audio_response.content)
    print(f"音声ファイルを保存しました: {len(audio_response.content)} バイト")

    # ffmpegでMP3に変換
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "2", output_path],
            check=True
        )
        print("ffmpegで変換完了")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg変換エラー: {e}")
        return "Audio processing failed", 500

    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
