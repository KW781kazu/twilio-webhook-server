from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/")
def index():
    return "Server is running", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    # Twilioからのリクエストを受け取る
    data = request.form
    print("Incoming data:", data)

    # 簡単な応答（電話用）
    response = '<?xml version="1.0" encoding="UTF-8"?><Response><Say>こんにちは。AI受付です。</Say></Response>'
    return response, 200, {'Content-Type': 'application/xml'}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

