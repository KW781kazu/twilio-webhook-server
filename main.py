from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    resp = VoiceResponse()
    # 日本語自然音声を指定
    with resp.gather(input='speech', language='ja-JP', timeout=5) as gather:
        gather.say("こんにちは。AI受付です。ご用件をお話しください。", language="ja-JP", voice="Polly.Mizuki")
    resp.say("もしもし。ご用件をお話しください。", language="ja-JP", voice="Polly.Mizuki")
    return Response(str(resp), mimetype="application/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


