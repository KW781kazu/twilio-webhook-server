from flask import Flask, request, Response
import os
import google.auth
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.cloud.aiplatform.gapic.schema import predict

app = Flask(__name__)

# 環境変数からプロジェクトIDを取得
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "gemini-pro"  # ここを安定版モデルに変更

# Vertex AI 初期化
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Twilio から受け取った音声のテキスト化（仮にここでは録音だけ）
        transcribed_text = "ユーザーが話した内容"  # ここはあとでSTT結果に差し替え
        
        # Geminiへ問い合わせ
        client = PredictionServiceClient()
        endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}"
        response = client.predict(
            endpoint=endpoint,
            instances=[{"content": transcribed_text}],
            parameters={"temperature": 0.2}
        )

        ai_response = response.predictions[0].get("content", "すみません、うまく処理できませんでした。")

        # TwiML 応答
        twiml = f"""
        <Response>
            <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
        </Response>
        """
        return Response(twiml, mimetype="text/xml")

    except Exception as e:
        print(f"Error: {e}")
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
