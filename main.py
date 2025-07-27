from flask import Flask, request, Response
import requests
import os
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
        recording_status_callback_event=["completed"]
    )
    return Response(str(resp), mimetype='application/xml')

@app.route("/recording", methods=['POST'])
def recording():
    """録音完了後の処理（まずはファイル取得だけ確認）"""
    recording_url = request.form.get('RecordingUrl') + ".wav"
    print(f"TwilioからのRecordingUrl: {recording_url}")

    try:
        # 音声ファイルをダウンロードして保存
        audio_content = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content
        temp_input = "/tmp/input_audio.wav"
        with open(temp_input, "wb") as f:
            f.write(audio_content)
        print(f"音声ファイルを保存しました: {temp_input}")
    except Exception as e:
        print(f"録音ファイル取得エラー: {e}")
        return Response("Error in recording fetch", status=500)

    return Response("OK", status=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
