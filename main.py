from flask import Flask, request, Response
import requests
import os
import openai
import time

app = Flask(__name__)

# 環境変数からAPIキーとTwilio認証情報を取得
openai.api_key = os.environ.get("OPENAI_API_KEY")
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

@app.route("/voice", methods=["POST"])
def voice():
    print("/voice にアクセスされました", flush=True)
    response = '''<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
        <Record maxLength="60" timeout="5" recordingFormat="mp3"
            recordingStatusCallback="https://twilio-webhook-server-3sq0.onrender.com/recording"/>
    </Response>
    '''
    return Response(response, mimetype='text/xml')

@app.route("/recording", methods=["POST"])
def recording():
    print("/recording にリクエストが来ました", flush=True)

    recording_url = request.form.get("RecordingUrl")
    print(f"TwilioからのRecordingUrl: {recording_url}", flush=True)

    if not recording_url:
        print("RecordingUrlが取得できませんでした", flush=True)
        return Response("NG", status=400)

    try:
        # 録音が完成するのを少し待つ
        time.sleep(2)

        # 認証付きでMP3を取得
        download_url = recording_url + ".mp3"
        print(f"ダウンロード対象URL: {download_url}", flush=True)
        audio_response = requests.get(download_url, auth=(account_sid, auth_token))

        if audio_response.status_code != 200:
            print(f"録音ファイル取得失敗: {audio_response.status_code}", flush=True)
            return Response("NG", status=500)

        audio_bytes = audio_response.content
        print(f"音声ファイルサイズ: {len(audio_bytes)} バイト", flush=True)

        # 一時ファイルに保存
        temp_path = "/tmp/audio.mp3"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        print("音声ファイルを一時保存しました", flush=True)

        # Whisperで文字起こし
        with open(temp_path, "rb") as f:
            response = openai.Audio.transcribe(model="whisper-1", file=f)
        transcript = response["text"]
        print(f"文字起こし結果: {transcript}", flush=True)

    except Exception as e:
        print(f"エラー発生: {e}", flush=True)

    return Response("OK", status=200)

@app.route("/", methods=["GET"])
def index():
    return "Twilio Webhook is running", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
