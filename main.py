from flask import Flask, request, Response
import requests
import os
import time
import ffmpeg

app = Flask(__name__)

TWILIO_AUTH = (os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN"))

@app.route("/voice", methods=["POST"])
def voice():
    response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
    <Record maxLength="60" timeout="5" recordingStatusCallback="/recording" />
</Response>"""
    return Response(response, mimetype="text/xml")

@app.route("/recording", methods=["POST"])
def recording():
    recording_url = request.form.get("RecordingUrl")
    recording_status = request.form.get("RecordingStatus")
    print(f"録音URL: {recording_url}")
    print(f"録音ステータス: {recording_status}")

    if recording_status != "completed":
        print("録音が完了していません。再試行をスキップ。")
        return Response("Recording not ready", status=200)

    try:
        # Twilio録音ファイルは拡張子なし → .wavでダウンロード
        audio_url = f"{recording_url}.wav"
        tmp_input = "/tmp/input_audio.wav"
        tmp_output = "/tmp/converted_audio.mp3"

        # ダウンロード（2秒待機して確実にファイルが準備されるように）
        time.sleep(2)
        print("音声ファイルをダウンロードします...")
        audio_data = requests.get(audio_url, auth=TWILIO_AUTH)
        with open(tmp_input, "wb") as f:
            f.write(audio_data.content)
        print(f"音声ファイルを保存しました: {tmp_input}, サイズ: {len(audio_data.content)} バイト")

        # ffmpegでMP3に変換
        print("音声をMP3に変換します...")
        ffmpeg.input(tmp_input).output(tmp_output, ar=44100, ac=2).overwrite_output().run()
        print(f"MP3変換完了: {tmp_output}")

        # ここでGeminiやAI処理に送る処理を書く
        print("（AI送信処理はここに追加予定）")

        return Response("録音処理完了", status=200)

    except Exception as e:
        print(f"録音処理でエラー: {e}")
        return Response("録音処理失敗", status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
