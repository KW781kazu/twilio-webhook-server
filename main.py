from flask import Flask, request, Response
import os
import json
from flask_sock import Sock
import base64
import audioop
from google.cloud import speech

app = Flask(__name__)
sock = Sock(app)

# Google Speech-to-Text クライアント
speech_client = speech.SpeechClient()

def get_streaming_config():
    return speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,  # Twilioは8kHz
            language_code="ja-JP",
        ),
        interim_results=True,  # 中間結果も取得
        single_utterance=False
    )

@app.route("/voice", methods=["POST"])
def voice():
    # Media Streams を開始
    response = f"""
    <Response>
        <Start>
            <Stream url="wss://{request.host}/media" />
        </Start>
        <Say language="ja-JP">ストリーミング音声認識のテストです。話してください。</Say>
        <Pause length="60"/>
    </Response>
    """
    return Response(response, mimetype="application/xml")

# WebSocketで音声を受信しGoogle STTに逐次送信
@sock.route('/media')
def media(ws):
    print("WebSocket: 接続開始")

    # Google STTとのストリーミング接続を作成
    requests_generator = streaming_request_generator()
    responses = speech_client.streaming_recognize(get_streaming_config(), requests_generator)

    # レスポンス処理を非同期で走らせる
    from threading import Thread
    def listen_responses():
        try:
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
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
                # μ-law → PCM16 変換
                payload = base64.b64decode(data["media"]["payload"])
                pcm_audio = audioop.ulaw2lin(payload, 2)
                # Google STTに送信
                audio_queue.append(pcm_audio)
            elif event == "start":
                print("WebSocket: Media stream started")
            elif event == "stop":
                print("WebSocket: Media stream stopped")
                break
        except Exception as e:
            print(f"WebSocket Error: {e}")
    print("WebSocket: 接続終了")

# Google STTへの音声送信用ジェネレーター
from collections import deque
audio_queue = deque()

def streaming_request_generator():
    while True:
        if audio_queue:
            chunk = audio_queue.popleft()
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
