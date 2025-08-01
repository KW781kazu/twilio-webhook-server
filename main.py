from flask import Flask, request, Response
import google.auth
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.cloud.aiplatform.gapic.schema.predict.instance import TextClassificationPredictionInstance
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os

app = Flask(__name__)

# 環境変数からプロジェクト情報を取得
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"  # Text-Bisonが利用可能なリージョン
MODEL_NAME = "text-bison"  # Text-Bisonを指定

# Vertex AI クライアント設定
client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
prediction_client = PredictionServiceClient(client_options=client_options)
model_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}"

def get_bison_response(text):
    instance = json_format.ParseDict({"content": text}, Value())
    response = prediction_client.predict(
        endpoint=model_path,
        instances=[instance],
        parameters=json_format.ParseDict({}, Value())
    )
    if response.predictions:
        return dict(response.predictions[0])["content"]
    else:
        return "すみません、うまく応答できませんでした。"

@app.route("/webhook", methods=["POST"])
def webhook():
    # Twilioからのリクエストを取得
    incoming_data = request.form
    speech_result = incoming_data.get("SpeechResult", "")

    # AI応答を取得
    ai_response = get_bison_response(f"次の内容に日本語で返答してください: {speech_result}")

    # Twilioに返答
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ja-JP" voice="Polly.Mizuki">{ai_response}</Say>
    <Gather input="speech" language="ja-JP" timeout="5" />
</Response>"""
    return Response(twiml_response, mimetype="text/xml")

@app.route("/", methods=["GET"])
def index():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
