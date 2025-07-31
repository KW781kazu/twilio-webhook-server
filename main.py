from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import os
import requests
import json

app = Flask(__name__)

# Gemini API設定
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Renderの環境変数で設定
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def get_gemini_response(user_text):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    data = {
        "contents": [{"parts": [{"text": user_text}]}]
    }
    print("Sending to Gemini:", data)
    response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(data))
    print("Gemini status:", response.status_code)
    print("Gemini response:", response.text)
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return "すみません、現在応答できません。"

@app.route("/webhook", methods=["POST"])
def webhook():
    user_input = request.form.get("SpeechResult", "")
    print("User said:", user_input)

    if user_input:
        ai_reply = get_gemini_response(user_input)
    else:
        ai_reply = "こんにちは。AI受付です。ご用件をお話しください。"

    resp = VoiceResponse()
    with resp.gather(input='speech', language='ja-JP', timeout=5) as gather:
        gather.say(ai_reply, language="ja-JP", voice="Polly.Mizuki")

    return Response(str(resp), mimetype="application/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
