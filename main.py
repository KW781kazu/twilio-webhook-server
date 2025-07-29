from flask import Flask, request, Response
import os
import json
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

@app.route("/voice", methods=["POST"])
def voice():
    # Media Streamsを開始するTwiML
    response = f"""
    <Response>
        <Start>
            <Stream url="wss://{request.host}/media" />
        </Start>
        <Say language="ja-JP">Media Streamsのテストです。話してください。</Say>
    </Response>
    """
    return Response(response, mimetype="application/xml")

# Twilioからの音声ストリームを受信
@sock.route('/media')
def media(ws):
    print("WebSocket: 接続開始")
    while True:
        message = ws.receive()
        if message is None:
            break
        try:
            data = json.loads(message)
            event = data.get("event")
            if event == "start":
                print("WebSocket: Media stream started")
            elif event == "media":
                payload = data["media"]["payload"]
                print(f"WebSocket: Audio payload {len(payload)} bytes")
            elif event == "stop":
                print("WebSocket: Media stream stopped")
                break
        except Exception as e:
            print(f"WebSocket Error: {e}")
    print("WebSocket: 接続終了")

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
