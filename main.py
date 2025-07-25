from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import requests
import os
import ffmpeg
import subprocess

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    resp.say("こんにちは。ご用件をどうぞ。", language="ja-JP")
    resp.record(
        max_length=60,
        timeout=5,
        recording_status_callback="/recording",
        recording_format="wav"
    )
    return Response(str(resp), mimetype="application/xml")

@app.route("/recording", methods=["POST"])
def recording():
    recording_url = request.form.get("RecordingUrl")
    if not recording_url:
        return "No recording URL", 400

    # 音声ファイル取得
    audio_url = f"{recording_url}.wav"
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    audio_response = requests.get(audio_url, auth=auth)
    input_path = "/tmp/input_audio.wav"
    output_path = "/tmp/converted_audio.mp3"
    with open(input_path, "wb") as f:
        f.write(audio_response.content)

    # ffmpegで変換
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "44100", "-ac", "2",
            output_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg変換エラー: {e}")
        return "Error processing audio", 500

    # GeminiやOpenAI APIの呼び出し（仮）
    print("音声処理完了:", output_path)

    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
