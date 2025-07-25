from flask import Flask, request, Response
import requests
import os
import subprocess
import io
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import speech

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Google Speech-to-Textクライアントの初期化
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/credentials.json"
speech_client = speech.SpeechClient()

@app.route("/voice", methods=['POST'])
def voice():
    """最初の応答: 録音を開始"""
    resp = VoiceResponse()
    resp.say("こんにちは。ご用件をどうぞ。", language="ja-JP")
    resp.record(
        max_length=60,
        timeout=5,
        play_beep=True,
        recording_status_callback="/recording"
    )
    return Response(str(resp), mimetype='application/xml')

@app.route("/recording", methods=['POST'])
def recording():
    """録音完了後の処理"""
    recording_url = request.form.get('RecordingUrl') + ".wav"  # WAV形式を取得
    print(f"TwilioからのRecordingUrl: {recording_url}")

    # 音声ファイルをダウンロード
    audio_content = requests.get(
        recording_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    ).content

    temp_input = "/tmp/input_audio.wav"
    temp_output = "/tmp/converted_audio.mp3"

    with open(temp_input, "wb") as f:
        f.write(audio_content)
    print(f"音声ファイルを保存しました: {temp_input}")

    # ffmpegでMP3に変換
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ar", "44100", "-ac", "2",
            temp_output
        ], check=True)
        print(f"音声ファイルを変換しました: {temp_output}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg変換エラー: {e}")
        return Response("Error in processing audio", status=500)

    # Google Speech-to-Textで文字起こし
    try:
        with io.open(temp_output, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=44100,
            language_code="ja-JP"
        )

        response = speech_client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        print(f"文字起こし結果: {transcript}")

    except Exception as e:
        print(f"Speech-to-Textエラー: {e}")
        return Response("Error in transcription", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
