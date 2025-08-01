from flask import Flask, request, Response
from google.cloud import aiplatform
import os

app = Flask(__name__)

# Vertex AI 初期化
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "text-bison@001"

aiplatform.init(project=PROJECT_ID, location=LOCATION)
model = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Twilio からの録音データはここで処理（今は仮テキスト）
        user_input = "こんにちは。"
        response_text = model.predict(user_input).text
        twiml = f"""
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">{response_text}</Say>
        </Response>
        """
        return Response(twiml, mimetype="application/xml")
    except Exception as e:
        print(f"Error: {e}")
        return Response("<Response><Say>現在お答えできません。</Say></Response>", mimetype="application/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
