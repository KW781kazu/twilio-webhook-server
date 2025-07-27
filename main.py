from flask import Flask, request, Response
import requests
import os
import subprocess
import time
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

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
        recording_status_callback_event=["completed"]  # 録音完了時のみ呼び出す
    )
    return Response(str(resp), mimetype='application/xml')

@app.route("/recording", methods=['POST'])
def recording():
    """録音完了後の処理"""
    recording_url = request.form.get('RecordingUrl') + ".wav"  # WAV形式を取得
    print(f"TwilioからのRecordingUrl: {recording_url}")

    # 取得リトライ（録音がまだ準備中の場合の対策）
    audio_content = None
    for i in range(3):  # 最大3回リトライ
        response = requests.get(recording_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        if response.status_code == 200 and len(response.content) > 1000:  # データがある程度あるか確認
            audio_content = response.content
            break
        print(f"録音ファイルが未準備。リトライ {i+1}/3")
        time.sleep(2)

    if audio_content is None:
        print("録音ファイル取得失敗。")
        return Response("Recording not ready", status=500)

    temp_input = "/tmp/input_audio.wav"
    temp_output = "/tmp/converted_audio.flac"

    with open(temp_input, "wb") as f:
        f.write(audio_content)
    print(f"音声ファイルを保存しました: {temp_input}")

    # ffmpegでFLACに変換（音声認識用）
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

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
