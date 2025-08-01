from flask import Flask, request, Response
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
import os
import traceback

app = Flask(__name__)

# 環境変数
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "text-bison@001"

# Prediction クライアント
client = PredictionServiceClient()
endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        print("=== Webhook called ===")
        print("Request form data:", request.form)

        # Twilioの音声認識結果
        user_input = request.form.get("SpeechResult", "")
        print("User input:", user_input)

        if not user_input:
            user_input = "何も聞き取れませんでした。"

        # Vertex AI にテキストを送信
        instance = {"content": user_input}
        parameters = {"temperature": 0.2, "maxOutputTokens": 256}

        response = client.predict(
            endpoint=endpoint,
            instances=[instance],
            parameters=parameters
        )
        print("Vertex AI response:", response)

        ai_response = response.predictions[0].get('content', "すみません、応答できませんでした。")

        # Twilio応答
        twiml_response = f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
            <Pause length="1"/>
            <Say language="ja-JP" voice="Polly.Mizuki">ご用件をお話しください。</Say>
        </Response>
        """
        return Response(twiml_response, mimetype="text/xml")

    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()
        error_response = """
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">現在お答えできません。後ほどおかけ直しください。</Say>
        </Response>
        """
        return Response(error_response, mimetype="text/xml")

@app.route("/")
def index():
    return "Twilio Webhook Server Running", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
