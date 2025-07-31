from flask import Flask, request, Response
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import os

app = Flask(__name__)

# Vertex AI 初期化
aiplatform.init(project=os.environ.get("GCP_PROJECT_ID"), location="us-central1")
gemini_model = GenerativeModel(
    "projects/{}/locations/us-central1/publishers/google/models/gemini-1.0-pro".format(
        os.environ.get("GCP_PROJECT_ID")
    )
)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        recording_url = request.form.get("RecordingUrl")
        if not recording_url:
            response = """<?xml version="1.0" encoding="UTF-8"?>
                <Response>
                    <Say language="ja-JP" voice="Polly.Mizuki">音声が取得できませんでした。もう一度お話しください。</Say>
                    <Record action="/webhook" maxLength="10" method="POST" playBeep="true" timeout="5" />
                </Response>"""
            return Response(response, mimetype="text/xml")

        # Gemini で応答生成
        prompt = f"以下の内容に自然な日本語で返答してください:\n録音URL: {recording_url}"
        result = gemini_model.generate_content(prompt)
        ai_response = result.candidates[0].content.parts[0].text if result.candidates else "すみません、内容を理解できませんでした。"

        response = f"""<?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
                <Record action="/webhook" maxLength="10" method="POST" playBeep="true" timeout="5" />
            </Response>"""
        return Response(response, mimetype="text/xml")
    except Exception as e:
        print(f"Error: {e}")
        response = """<?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Say language="ja-JP" voice="Polly.Mizuki">現在応答できません。後ほどおかけ直しください。</Say>
            </Response>"""
        return Response(response, mimetype="text/xml")

@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
