from flask import Flask, request, Response
import requests
import os
import subprocess
from google.cloud import speech
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

# Twilio認証情報
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Google API認証情報
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/credentials.json"

# RenderのURL（RecordingStatusCallback用にフルパスで使用）
BASE_URL = "https://twilio-webhook-server-3sq0.onrender.com"

@app.route("/voice", methods=['POST'])
def voice():
    """最初の応答: 録音を開始"""
    resp = VoiceResponse()
    resp.say("こんにちは。ご用件をどうぞ。", language="ja-JP")
    # RecordingStatusCallback をフルURLに修正
    resp.record(
        max_length=60,
        timeout=5,
        play_beep=True,
        recording_status_callback=f"{BASE_URL}/recording",
        recording_status_callback_event=["completed"]
    )
    print("[DEBUG] /voice エンドポイントが呼ばれました")
    return Response(str(resp), mimetype='application/xml')

@app.route("/recording", methods=['POST'])
def recording():
    """録音完了後の処理: 音声を取得して文字起こし"""
    print("[DEBUG] /recording エンドポイントに到達")
    print("[DEBUG] 受け取ったデータ:", request.form.to_dict())

    recording_url = request.form.get('RecordingUrl') + ".wav"
    print(f"[INFO] TwilioからのRecordingUrl: {recording_url}")

    # 音声ファイルをダウンロード
    try:
        audio_content = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content
    except Exception as e:
        print(f"[ERROR] 音声ファイルのダウンロードエラー: {e}")
        return Response("Error downloading audio", status=500)

    temp_input = "/tmp/input_audio.wav"
    temp_output = "/tmp/converted_audio.flac"

    with open(temp_input, "wb") as f:
        f.write(audio_content)
    print(f"[INFO] 音声ファイルを保存しました: {temp_input}")

    # ffmpegでFLACに変換（16kHzにリサンプリング）
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ar", "16000", "-ac", "1",
            temp_output
        ], check=True)
        print(f"[INFO] 音声ファイルを変換しました: {temp_output}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg変換エラー: {e}")
        return Response("Error in processing audio", status=500)

    # Google Speech-to-Text
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
        print(f"[DEBUG] Google Speech API response: {response}")
        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript
        print(f"[INFO] 文字起こし結果: {transcription}")
    except Exception as e:
        print(f"[ERROR] Google Speech-to-Textエラー: {e}")
        transcription = ""

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
