from flask import Flask, request, Response
import os
import json
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

@app.route("/voice", methods=["POST"])
def voice():
    # Media Streamsを開始するTwiMLを返す
    response = f"""
    <Response>
        <Start>
            <Stream url="wss://{request.host}/media" />
        </Start>
        <Say language="ja-JP">こんにちは。ご用件をどうぞ。</Say>
    </Response>
    """
    return Response(response, mimetype="application/xml")

# WebSocketでTwilio音声を受け取る
@sock.route('/media')
def media(ws):
    while True:
        message = ws.receive()
        if message is None:
            break
        data = json.loads(message)
        event = data.get("event")
        if event == "media":
            payload = data["media"]["payload"]
            print(f"Audio payload received: {len(payload)} bytes")
        elif event == "start":
            print("Media stream started")
        elif event == "stop":
            print("Media stream stopped")
            break

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
