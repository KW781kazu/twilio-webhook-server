from flask import Flask, request, Response
import requests
import os
import subprocess
import logging
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import speech

# ログ設定
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Twilio認証情報
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Google Speech-to-Text クライアント
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/credentials.json"
speech_client = speech.SpeechClient()

@app.route("/voice", methods=['POST'])
def voice():
    """最初の応答: 録音を開始"""
    logging.info("音声通話開始: /voice にアクセス")
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
    recording_url = request.form.get('RecordingUrl') + ".wav"  # WAV形式を取得
    logging.info(f"録音完了: {recording_url}")

    # 音声ファイルをダウンロード
    try:
        audio_content = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content
        temp_input = "/tmp/input_audio.wav"
        with open(temp_input, "wb") as f:
            f.write(audio_content)
        logging.info(f"音声ファイル保存: {temp_input}")
    except Exception as e:
        logging.error(f"音声ダウンロード失敗: {e}")
        return Response("Audio download error", status=500)

    # 音声を FLAC に変換（Google Speech-to-Text用）
    temp_output = "/tmp/converted_audio.flac"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ar", "16000", "-ac", "1",
            temp_output
        ], check=True)
        logging.info(f"音声変換成功: {temp_output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg変換エラー: {e}")
        return Response("Audio conversion error", status=500)

    # Google Speech-to-Text で認識
    try:
        with open(temp_output, "rb") as f:
            audio_data = f.read()
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="ja-JP"
        )
        response = speech_client.recognize(config=config, audio=audio)
        text = ""
        for result in response.results:
            text += result.alternatives[0].transcript
        logging.info(f"認識結果: {text}")
    except Exception as e:
        logging.error(f"音声認識エラー: {e}")
        return Response("Speech-to-Text error", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
