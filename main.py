from flask import Flask, request, Response
from google.cloud import aiplatform
import os

app = Flask(__name__)

# プロジェクト情報
PROJECT_ID = "cloudrun-demo-20250701"  # あなたのプロジェクトID
LOCATION = "us-central1"
MODEL_NAME = "text-bison@002"  # 修正版

# Vertex AI 初期化
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# モデルをロード
model = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        user_input = "こんにちは。お名前を教えてください。"  # 後でSTTに置き換える

        # Vertex AI モデル呼び出し
        response = model.predict(
            user_input,
            temperature=0.2,
            max_output_tokens=256
        )

        ai_response = response.text if response.text else "すみません、うまく応答できませんでした。"

        # Twilio用レスポンス
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
