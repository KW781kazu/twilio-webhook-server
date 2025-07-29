from flask import Flask, request, Response
import os
import json
from flask_sock import Sock
import base64
from pydub import AudioSegment
from google.cloud import speech
from io import BytesIO
from collections import deque
from threading import Thread

app = Flask(__name__)
sock = Sock(app)

speech_client = speech.SpeechClient()
audio_queue = deque()

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

# μ-law を PCM16 に変換
def ulaw_to_pcm16(ulaw_bytes):
    audio = AudioSegment(
        data=ulaw_bytes,
        sample_width=1,  # μ-law = 8bit
        frame_rate=8000,
        channels=1
    )
    pcm_wav = BytesIO()
    audio.export(pcm_wav, format="wav")
    return pcm_wav.getvalue()

@sock.route('/media')
def media(ws):
    print("WebSocket: 接続開始")

    requests_generator = streaming_request_generator()
    responses = speech_client.streaming_recognize(get_streaming_config(), requests_generator)

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

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
