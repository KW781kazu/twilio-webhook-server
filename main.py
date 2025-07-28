from flask import Flask, request, Response
import requests
import os
import subprocess
import traceback
from google.cloud import speech
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

# Twilio認証情報
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Google API認証情報
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
    """録音完了後の処理: 音声を取得して文字起こし"""
    recording_url = request.form.get('RecordingUrl') + ".wav"
    print(f"[INFO] TwilioからのRecordingUrl: {recording_url}")

    # 音声ファイルをダウンロード
    try:
        audio_content = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content
        print("[INFO] 音声ファイルをダウンロードしました")
    except Exception as e:
        print(f"[ERROR] 音声ファイルのダウンロードに失敗: {e}")
        traceback.print_exc()
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
        traceback.print_exc()
        return Response("Error in processing audio", status=500)

    # Google Speech-to-Text
    transcription = ""
    try:
        print("[INFO] Google Speech-to-Text API呼び出し開始")
        client = speech.SpeechClient()
        with open(temp_output, "rb") as audio_file:
            audio = speech.RecognitionAudio(content=audio_file.read())

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="ja-JP"
        )

        response = client.recognize(config=config, audio=audio)
        for result in response.results:
            transcription += result.alternatives[0].transcript
        print(f"[INFO] 文字起こし結果: {transcription}")
    except Exception as e:
        print(f"[ERROR] Google Speech-to-Textエラー: {e}")
        traceback.print_exc()
        transcription = "音声を認識できませんでした"

    # Twilioに文字起こし結果を返答
    resp = VoiceResponse()
    if transcription:
        resp.say(f"認識結果は: {transcription}", language="ja-JP")
    else:
        resp.say("音声を認識できませんでした。", language="ja-JP")

    return Response(str(resp), mimetype='application/xml')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
