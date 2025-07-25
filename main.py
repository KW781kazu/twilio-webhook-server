from flask import Flask, request, Response
import requests
import os
import openai
import subprocess

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route("/voice", methods=["POST"])
def voice():
    print("📞 /voice にアクセスされました", flush=True)
    response = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
    <Record 
        maxLength="60" 
        timeout="5" 
        recordingStatusCallback="/recording" />
</Response>
'''
    return Response(response, mimetype='text/xml')

@app.route("/recording", methods=["POST"])
def recording():
    print("🎙 /recording にリクエストが来ました", flush=True)
    recording_url = request.form.get("RecordingUrl")
    print(f"TwilioからのRecordingUrl: {recording_url}", flush=True)

    if not recording_url:
        print("❌ RecordingUrlが受け取れていません", flush=True)
        return Response("NG", status=400)

    try:
        # Twilioの録音ファイルを取得（拡張子なし＝WAV）
        audio_response = requests.get(recording_url + ".wav")
        audio_bytes = audio_response.content
        print(f"音声ファイルを {len(audio_bytes)} バイト取得しました", flush=True)

        # 一時ファイルに保存
        input_path = "/tmp/input.wav"
        with open(input_path, "wb") as f:
            f.write(audio_bytes)
        print("一時ファイルとして保存しました", flush=True)

        # Whisper API で文字起こし
        with open(input_path, "rb") as f:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=f
            )
        transcript = response["text"]
        print(f"文字起こし結果: {transcript}", flush=True)

    except Exception as e:
        print(f"🔥 エラー発生: {e}", flush=True)
        return Response("NG", status=500)

    return Response("OK", status=200)

@app.route("/", methods=["GET"])
def index():
    return "Twilio Webhook is running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
