from flask import Flask, request, Response
import requests
import os
import subprocess
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import speech

# Flask 初期化
app = Flask(__name__)

# Twilio 認証
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Google 認証ファイルのパスを強制設定
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
    print("録音処理開始 /recording にアクセス")

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

    # ffmpegでFLACに変換（Google Speech-to-Text推奨）
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

    # Google Speech-to-Textで文字起こし
    try:
        client = speech.SpeechClient()
        with open(temp_output, "rb") as audio_file:
            audio_data = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="ja-JP"
        )
        response = client.recognize(config=config, audio=audio)
        print("Speech-to-Text API 呼び出し成功")

        for result in response.results:
            print(f"Transcription result: {result.alternatives[0].transcript}")

    except Exception as e:
        print(f"Speech-to-Text エラー: {e}")
        return Response("Error in transcription", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
