from flask import Flask, request, Response
import requests
import os
import subprocess
from google.cloud import speech
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/credentials.json"

@app.route("/voice", methods=['POST'])
def voice():
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
    recording_url = request.form.get('RecordingUrl') + ".wav"
    print(f"[INFO] Twilio Recording URL: {recording_url}")

    # 音声ダウンロード
    try:
        audio_content = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content
    except Exception as e:
        print(f"[ERROR] 音声ダウンロード失敗: {e}")
        return Response("Error downloading audio", status=500)

    temp_input = "/tmp/input_audio.wav"
    temp_output = "/tmp/converted_audio.flac"

    with open(temp_input, "wb") as f:
        f.write(audio_content)
    print(f"[INFO] 音声保存: {temp_input}")

    # ffmpeg変換
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ar", "16000", "-ac", "1",
            temp_output
        ], check=True)
        print(f"[INFO] 変換後ファイル: {temp_output}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg変換エラー: {e}")
        return Response("Error processing audio", status=500)

    # Google Speech-to-Text
    try:
        print("[INFO] Google Speech-to-Text 呼び出し開始")
        client = speech.SpeechClient()
        with open(temp_output, "rb") as audio_file:
            audio = speech.RecognitionAudio(content=audio_file.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="ja-JP"
        )
        response = client.recognize(config=config, audio=audio)
        transcription = "".join([result.alternatives[0].transcript for result in response.results])
        print(f"[INFO] 文字起こし結果: {transcription}")
    except Exception as e:
        print(f"[ERROR] Google Speech-to-Text エラー: {e}")
        transcription = ""

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
