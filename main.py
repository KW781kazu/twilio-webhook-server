from flask import Flask, request, Response
import os
from google.cloud import aiplatform

app = Flask(__name__)

# プロジェクトとリージョンを環境変数から取得
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"  # リージョンを明示的に設定

# Vertex AI 初期化
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# モデル名（最低限動作確認のため text-bison に変更）
MODEL_NAME = "text-bison"

# モデルを取得
model = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.form
        transcription = data.get("SpeechResult", "")
        print(f"Received transcription: {transcription}")

        # Gemini呼び出し（ここでは bison で応答を確認）
        response = model.predict(
            prompt=f"ユーザーの発話: {transcription}\n自然な日本語で返答してください。",
            temperature=0.2,
            max_output_tokens=256,
        )

        reply_text = response.text.strip()
        print(f"Model response: {reply_text}")

        twiml_response = f"""
            <Response>
                <Say language="ja-JP" voice="Polly.Mizuki">{reply_text}</Say>
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
