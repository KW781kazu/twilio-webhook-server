from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    resp = VoiceResponse()
    # 日本語音声 + しばらく待機
    with resp.gather(input='speech', language='ja-JP', timeout=5) as gather:
        gather.say("こんにちは。AI受付です。ご用件をお話しください。", language="ja-JP")
    # 何も応答がなければ再度話す
    resp.say("もしもし。ご用件をお話しください。", language="ja-JP")
    return Response(str(resp), mimetype="application/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


