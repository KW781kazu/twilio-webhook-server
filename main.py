from flask import Flask, request, Response
import os
import json
from flask_sock import Sock
import base64
from google.cloud import speech
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from collections import deque
from threading import Thread

app = Flask(__name__)
sock = Sock(app)

# Google Speech-to-Text
speech_client = speech.SpeechClient()

# Vertex AI + Gemini 設定
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = "us-central1"  # Gemini対応リージョン
vertexai.init(project=PROJECT_ID, location=LOCATION)
gemini_model = GenerativeModel("gemini-1.5-flash")

audio_queue = deque()
latest_transcript = ""  # 最新の認識結果を保持

def get_streaming_config():
    return speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,
            language_code="ja-JP",
        ),
        interim_results=True,
        single_utterance=False
    )

@app.route("/voice", methods=["POST"])
def voice():
    global latest_transcript
    user_text = latest_transcript if latest_transcript else "こんにちは"
    ai_response = get_gemini_response(user_text)
    print(f"[Gemini応答] {ai_response}")

    # Twilioに返答
    response = f"""
    <Response>
        <Say language="ja-JP">{ai_response}</Say>
        <Pause length="5"/>
    </Response>
    """
    return Response(response, mimetype="application/xml")

# μ-law デコード（軽量版）
def ulaw_decode(byte):
    MU = 255
    BIAS = 132
    u_val = ~byte & 0xFF
    t = ((u_val & 0x0F) << 3) + BIAS
    t <<= (u_val & 0x70) >> 4
    return -t if (u_val & 0x80) else t - BIAS

def ulaw_to_pcm16(ulaw_bytes):
    pcm16 = bytearray()
    for b in ulaw_bytes:
        sample = ulaw_decode(b)
        pcm16 += sample.to_bytes(2, byteorder='little', signed=True)
    return bytes(pcm16)

@sock.route('/media')
def media(ws):
    global latest_transcript
    print("WebSocket: 接続開始")

    requests_generator = streaming_request_generator()
    responses = speech_client.streaming_recognize(get_streaming_config(), requests_generator)

    def listen_responses():
        global latest_transcript
        try:
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    latest_transcript = transcript
                    if result.is_final:
                        print(f"[確定] {transcript}")
                    else:
                        print(f"[暫定] {transcript}")
        except Exception as e:
            print(f"Google STT Error: {e}")

    Thread(target=listen_responses, daemon=True).start()

    while True:
        message = ws.receive()
        if message is None:
            break
        try:
            data = json.loads(message)
            event = data.get("event")
            if event == "media":
                payload = base64.b64decode(data["media"]["payload"])
                pcm_audio = ulaw_to_pcm16(payload)
                audio_queue.append(pcm_audio)
            elif event == "start":
                print("WebSocket: Media stream started")
            elif event == "stop":
                print("WebSocket: Media stream stopped")
                break
        except Exception as e:
            print(f"WebSocket Error: {e}")
    print("WebSocket: 接続終了")

def streaming_request_generator():
    while True:
        if audio_queue:
            chunk = audio_queue.popleft()
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

# Gemini応答生成（Vertex AI）
def get_gemini_response(text):
    try:
        response = gemini_model.generate_content(f"次の発話に自然に返答してください：{text}")
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "すみません、今は応答できません。"

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
