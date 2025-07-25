from flask import Flask, request, Response
import requests
import os
import openai

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route("/voice", methods=["POST"])
def voice():
    print("📞 /voice にアクセスされました", flush=True)
    response = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
    <Record maxLength="60" timeout="5" recordingStatusCallback="/recording" />
</Response>
'''
    return Response(response, mimetype='text/xml')

@app.route("/recording", methods=["POST"])
def recording():
    print("📥 /recording にリクエストが来ました", flush=True)
    recording_url = request.form.get("RecordingUrl")
    print(f"TwilioからのRecordingUrl: {recording_url}", flush=True)

    transcript = ""
    if recording_url:
        try:
            audio_response = requests.get(recording_url)
            audio_bytes = audio_response.content
            temp_path = "/tmp/audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            with open(temp_path, "rb") as f:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=f
                )
            transcript = response["text"]
            print(f"📝 文字起こし結果: {transcript}", flush=True)
        except Exception as e:
            print(f"🔥 エラー発生: {e}", flush=True)

    # 録音後の応答を返す
    response = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP">ありがとうございました。担当者におつなぎします。</Say>
    <Hangup/>
</Response>
'''
    return Response(response, mimetype='text/xml')

@app.route("/", methods=["GET"])
def index():
    return "Twilio Webhook is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
