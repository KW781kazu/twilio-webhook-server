from flask import Flask, request, Response
import requests
import os
import subprocess
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
        recording_status_callback="/recording"
    )
    return Response(str(resp), mimetype='application/xml')

@app.route("/recording", methods=['POST'])
def recording():
    """録音完了後の処理"""
    recording_url = request.form.get('RecordingUrl') + ".wav"
    print(f"[INFO] TwilioからのRecordingUrl: {recording_url}")

    try:
        # 音声ファイルダウンロード
        audio_content = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content
        temp_input = "/tmp/input_audio.wav"
        temp_output = "/tmp/converted_audio.flac"

        with open(temp_input, "wb") as f:
            f.write(audio_content)
        print(f"[INFO] 音声ファイル保存: {temp_input}")

        # ffmpegでFLACに変換（STT向け）
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input,
            "-ar", "16000", "-ac", "1", "-f", "flac",
            temp_output
        ], check=True)
        print(f"[INFO] 音声ファイル変換成功: {temp_output}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg変換エラー: {e}")
        return Response("Error in audio conversion", status=500)
    except Exception as e:
        print(f"[ERROR] 音声処理エラー: {e}")
        return Response("Error in recording processing", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
