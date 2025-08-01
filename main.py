from flask import Flask, request, Response
import os
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient

app = Flask(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "text-bison"  # 一旦無料で確実なモデルに

aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        transcribed_text = "ユーザーが話した内容"

        client = PredictionServiceClient()
        endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}"
        response = client.predict(
            endpoint=endpoint,
            instances=[{"content": transcribed_text}],
            parameters={"temperature": 0.2}
        )

        ai_response = response.predictions[0].get("content", "すみません、うまく処理できませんでした。")
        print(f"AI response: {ai_response}")

        twiml = f"""
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
        </Response>
        """
        return Response(twiml, mimetype="text/xml")

    except Exception as e:
        import traceback
        print("=== ERROR OCCURRED ===")
        print(e)
        traceback.print_exc()

        error_twiml = """
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">現在お答えできません。後ほどおかけ直しください。</Say>
        </Response>
        """
        return Response(error_twiml, mimetype="text/xml")

@app.route("/", methods=["GET"])
def index():
    return "Webhook is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
