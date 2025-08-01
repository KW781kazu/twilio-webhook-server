from flask import Flask, request, Response
from google.cloud import aiplatform
import os

app = Flask(__name__)

# 環境変数からプロジェクトとリージョンを取得
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cloudrun-demo-20250701")
LOCATION = os.getenv("GCP_REGION", "us-central1")
MODEL_NAME = "gemini-1.0-pro"  # 基本モデルで動作確認

# Vertex AI 初期化
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.route("/webhook", methods=["POST"])
def webhook():
    """Twilioからの音声→テキストを受け取り、Vertex AIで応答生成"""
    try:
        user_input = request.form.get("SpeechResult", "")
        if not user_input:
            return twiml_response("音声が認識できませんでした。もう一度お話しください。")

        # Vertex AI GenerativeModel を使用
        model = aiplatform.GenerativeModel(MODEL_NAME)
        response = model.generate_content([f"次の内容に自然な日本語で返答してください: {user_input}"])
        reply_text = response.text if response and hasattr(response, "text") else "現在お答えできません。"

        return twiml_response(reply_text)
    except Exception as e:
        print(f"Error: {e}")
        return twiml_response("現在お答えできません。後ほどおかけ直しください。")

def twiml_response(text: str):
    """Twilioに返すXMLレスポンス"""
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP" voice="Polly.Mizuki">{text}</Say>
    <Gather input="speech" language="ja-JP" timeout="5" />
</Response>"""
    return Response(twiml, mimetype="text/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
