from flask import Flask, request, Response
import os

app = Flask(__name__)

@app.route("/voice", methods=["POST"])
def voice():
    # 通常のTwiML応答だけ返す
    response = """
    <Response>
        <Say language="ja-JP">テストです。通話ができています。</Say>
        <Pause length="1"/>
        <Say language="ja-JP">このメッセージが聞こえたら接続は成功です。</Say>
        <Hangup/>
    </Response>
    """
    return Response(response, mimetype="application/xml")

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
