from flask import Flask, request, jsonify, make_response, send_file, send_from_directory
from TTS.api import TTS
from flask_cors import CORS
from datetime import timedelta
import soundfile as sf
import io
import whisper
import os
import zipfile

app = Flask(__name__)

CORS(app, resources={
    r'/*': {
        'origins': ['http://localhost:4000'],
        'allow_headers': ['Content-Type', 'Authorization'],
        'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        'supports_credentials': True,
        'allow_credentials': True
    }
}, send_wildcard=False)


@app.route("/")
def home():
    return "Hello, World!"

@app.route("/tts", methods=["POST","GET"])
def tts():
    if os.path.exists('output.zip'):
        os.remove('output.zip')
    inp = request.get_json()
    text = inp['text']
    speed = inp['speed']
    if speed=='medium':
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=False, gpu=False)
    elif speed=='fast':
        tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
    elif speed=='slow':
        tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
    else:
        return jsonify({"error": "Unknown voice model"}), 400
    audio=tts.tts(text=text)
    tts.tts_to_file(text, file_path="sound.wav")
    bytes_io = io.BytesIO()
    sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
    bytes_io.seek(0)

    model = whisper.load_model("medium") # Change this to your desired model
    transcribe = model.transcribe('sound.wav',fp16=False)
    segments = transcribe['segments']

    with zipfile.ZipFile('output.zip', 'w') as zipFile:
        for segment in segments:
            startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
            text = segment['text']
            segmentId = segment['id']+1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"

            with open('sub.srt', 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)

            zipFile.write('sub.srt')
            zipFile.write('sound.wav')
    if os.path.exists('sub.srt'):
        os.remove('sub.srt')
    if os.path.exists('sound.wav'):
        os.remove('sound.wav')
    return send_file('output.zip', as_attachment=True)


    # directory = os.getcwd()
    
    # audiopath='output.wav'
    # subpath='output.srt'
    # return send_from_directory(directory, audiopath), send_from_directory(directory, subpath)
    #return send_file(bytes_io, mimetype='audio/wav', as_attachment=True, download_name='output.wav')
    

    # audio_file = "output.wav"
    # return send_file(audio_file, attachment_filename=os.path.basename(audio_file), as_attachment=True)
    # audio = np.array(audio)  # Convert list to NumPy array
    # response = make_response(audio.tobytes())
    # response.headers.set('Content-Type', 'audio/wav')
    # response.headers.set('Content-Disposition', 'attachment', filename='output.wav')
    # return response
    # Run TTS
    # text = request.json["text"]
    # lang = request.json["lang"]
    # voice = request.json["voice"]
    # if voice == "tts_model":
    #     audio = tts_service.synthesize(text, lang=lang)
    # elif voice == "ljspeech_tts_model":
    #     audio = tts_service.synthesize(text, lang=lang, voice_model_path="tts_models/ljspeech/ljspeech.pth.tar")
    # elif voice == "ljspeech_fast_tts_model":
    #     audio = tts_service.synthesize(text, lang=lang, voice_model_path="tts_models/ljspeech/ljspeech_fast.pth.tar")
    # elif voice == "libritts_tts_model":
    #     audio = tts_service.synthesize(text, lang=lang, voice_model_path="tts_models/libritts/libritts.pth.tar")
    # else:
    #     return jsonify({"error": "Unknown voice model"}), 400
    #return "Flask server"
    #return jsonify({"audio": audio.tolist()}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)