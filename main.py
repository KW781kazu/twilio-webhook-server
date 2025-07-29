from flask import Flask, request, Response
import requests
import os
from google.cloud import speech

app = Flask(__name__)

# Twilio認証情報
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# 会話状態管理（CallSidごと）
conversation_state = {}

@app.route("/voice", methods=["POST"])
def voice():
    call_sid = request.form.get("CallSid")
    conversation_state[call_sid] = {"step": 1, "data": {}}
    response = """
    <Response>
        <Say language="ja-JP">こんにちは。お名前を教えてください。</Say>
        <Record action="/recording" method="POST" maxLength="30" playBeep="true"/>
    </Response>
    """
    return Response(response, mimetype="application/xml")

@app.route("/recording", methods=["POST"])
def process_recording():
    try:
        call_sid = request.form.get("CallSid")
        state = conversation_state.get(call_sid, {"step": 1, "data": {}})
        step = state["step"]

        recording_url = request.form.get("RecordingUrl")
        audio_response = requests.get(
            f"{recording_url}.wav",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        audio_content = audio_response.content

        # 音声認識
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="ja-JP"
        )
        response = client.recognize(config=config, audio=audio)
        transcript = response.results[0].alternatives[0].transcript if response.results else ""

        # 会話分岐
        if step == 1:
            state["data"]["name"] = transcript
            state["step"] = 2
            reply = f"""
            <Response>
                <Say language="ja-JP">ありがとうございます、{transcript}さん。車種を教えてください。</Say>
                <Record action="/recording" method="POST" maxLength="30" playBeep="true"/>
            </Response>
            """
        elif step == 2:
            state["data"]["car"] = transcript
            state["step"] = 3
            reply = f"""
            <Response>
                <Say language="ja-JP">ありがとうございます。{state['data']['name']}さんの{transcript}ですね。これで受付を完了しました。</Say>
                <Hangup/>
            </Response>
            """
        else:
            reply = "<Response><Say language='ja-JP'>ありがとうございました。</Say><Hangup/></Response>"

        conversation_state[call_sid] = state
        return Response(reply, mimetype="application/xml")

    except Exception as e:
        error_reply = f"<Response><Say>エラーが発生しました: {str(e)}</Say></Response>"
        return Response(error_reply, mimetype="application/xml")

@app.route("/")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
