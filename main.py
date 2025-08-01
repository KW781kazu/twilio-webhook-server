from flask import Flask, request, Response
from google.cloud import aiplatform
import os

app = Flask(__name__)

# プロジェクト情報
PROJECT_ID = "cloudrun-demo-20250701"  # あなたのプロジェクトIDに置き換え
LOCATION = "us-central1"
MODEL_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/text-bison@002"

# Vertex AI 初期化
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Webhookエンドポイント
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Twilio から録音データのテキスト（仮に user_input として受け取る）
        user_input = "こんにちは。お名前を教えてください。"  # ここはSTT処理で置き換え予定

        # Vertex AI モデル呼び出し
        client = aiplatform.gapic.PredictionServiceClient()
        endpoint = MODEL_NAME
        instances = [{"prompt": user_input}]

        response = client.predict(
            endpoint=endpoint,
            instances=instances
        )

        ai_response = response.predictions[0].get("content", "すみません、うまく応答できませんでした。")

        # Twilio 用レスポンス
        twiml_response = f"""
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
        </Response>
        """
        return Response(twiml_response, mimetype="text/xml")

    except Exception as e:
        print(f"Error: {e}")
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
