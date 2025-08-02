from flask import Flask, request, Response
from google.cloud import aiplatform
import os
import traceback

app = Flask(__name__)

# プロジェクト情報
PROJECT_ID = "cloudrun-demo-20250701"
LOCATION = "us-central1"
MODEL_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/text-bison@002"

# Vertex AI 初期化
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        user_input = "こんにちは。お名前を教えてください。"

        # Vertex AI モデル呼び出し
        print(f"Calling Vertex AI model: {MODEL_NAME}")
        client = aiplatform.gapic.PredictionServiceClient()
        endpoint = MODEL_NAME
        instances = [{"prompt": user_input}]
        print(f"Instances: {instances}")

        response = client.predict(
            endpoint=endpoint,
            instances=instances
        )

        print(f"Vertex AI response: {response}")

        ai_response = ""
        if hasattr(response, "predictions") and len(response.predictions) > 0:
            ai_response = response.predictions[0].get("content", "")
        if not ai_response:
            ai_response = "すみません、うまく応答できませんでした。"

        twiml_response = f"""
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
        </Response>
        """
        return Response(twiml_response, mimetype="text/xml")

    except Exception as e:
        print(f"Error in webhook: {e}")
        traceback.print_exc()
        error_response = """
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">現在お答えできません。後ほどおかけ直しください。</Say>
        </Response>
        """
        return Response(error_response, mimetype="text/xml")

@app.route("/")
def index():
    return "Twilio Webhook Server is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
