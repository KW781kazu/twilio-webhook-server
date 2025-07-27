from flask import Flask, request, Response
import requests
import os
import subprocess
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import speech

app = Flask(__name__)

# Twilio 認証情報
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Google 認証ファイル
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/credentials.json"

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

    if not audio_content:
        print("録音ファイルが空です。")
        return Response("No audio content", status=400)

    temp_input = "/tmp/input_audio.wav"
    temp_output = "/tmp/converted_audio.flac"

    with open(temp_input, "wb") as f:
        f.write(audio_content)
    print(f"音声ファイルを保存しました: {temp_input}")

    # ffmpegでFLACに変換
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ac", "1", "-ar", "16000",
            temp_output
        ], check=True)
        print(f"音声ファイルを変換しました: {temp_output}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg変換エラー: {e}")
        return Response("Error in processing audio", status=500)

    # Google Speech-to-Text で文字起こし
    try:
        client = speech.SpeechClient()
        with open(temp_output, "rb") as audio_file:
            audio = speech.RecognitionAudio(content=audio_file.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="ja-JP"
        )
        response = client.recognize(config=config, audio=audio)
        transcript = "".join([result.alternatives[0].transcript for result in response.results])
        print(f"文字起こし結果: {transcript}")
    except Exception as e:
        print(f"Speech-to-Text エラー: {e}")
        return Response("Error in speech-to-text", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
