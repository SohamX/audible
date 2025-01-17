import base64
from flask import Flask, request, jsonify, make_response, send_file, send_from_directory
from TTS.api import TTS
from flask_cors import CORS
from datetime import timedelta
import soundfile as sf
import io
import whisper
import os
import zipfile
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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

@app.route("/tts", methods=["POST"])
def tts():
    inp = request.get_json()
    text = inp['text']
    if(text[0]=='"'):
        text=text[1:]
    if(text[-1]=='"'):
        text=text[:-1]

    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('’', '')
    text = text.replace('‘', '')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("\r", '')
    text = text.replace("\n", '')
    speed = inp['speed']
    lan = inp['lan']
    print(text)
    # if(lan=='en'):
    if (speed=='medium' and lan=='en'):
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=False, gpu=False)
        # Split the input text into chunks of maximum length 100 characters
        text_chunks = [text[i:i+100] for i in range(0, len(text), 100)]
        
        audio_chunks = []
        stock=time.time()
        for i, chunk in enumerate(text_chunks):
            audio_chunk = tts.tts(text=chunk)
            audio_chunks.append(audio_chunk)
            tts.tts_to_file(chunk, file_path=f"sound_{i}.wav")
        ond=time.time()
        print(ond-stock,"for tts")
        bytes_io = io.BytesIO()
        audio = np.concatenate(audio_chunks, axis=None)
        sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
        bytes_io.seek(0)
        # Write the audio to a file
        with open('sound.wav', 'wb') as f:
            f.write(bytes_io.read())
        
        for i in range(len(text_chunks)):
            if os.path.exists(f"sound_{i}.wav"):
                os.remove(f"sound_{i}.wav") 

    elif (speed=='fast' and lan=='en'):
        tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
        stock=time.time()
        audio=tts.tts(text=text)
        tts.tts_to_file(text, file_path="sound.wav")
        ond=time.time()
        print(ond-stock,"for tts")
        bytes_io = io.BytesIO()
        sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
        bytes_io.seek(0)
    elif (speed=='slow' and lan=='en'):

        tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
        stock=time.time()
        audio=tts.tts(text=text)
        tts.tts_to_file(text, file_path="sound.wav")
        ond=time.time()
        print(ond-stock,"for tts")
        bytes_io = io.BytesIO()
        sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
        bytes_io.seek(0)
    elif(lan=='ga'):
        try:
            tts = TTS(model_name="tts_models/ga/cv/vits", progress_bar=False, gpu=False)
            stock=time.time()
            audio=tts.tts(text=text)
            tts.tts_to_file(text, file_path="sound.wav")
            ond=time.time()
            print(ond-stock,"for tts")
            bytes_io = io.BytesIO()
            sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
            bytes_io.seek(0)
        except ZeroDivisionError as e:
            print("Error:", e)
        
    else:
        return jsonify({"error": "Unknown voice model"}), 400

    # else:
    #     return jsonify({"error": "Unknown language"}), 400
    # audio=tts.tts(text=text)
    # tts.tts_to_file(text, file_path="sound.wav")
    # bytes_io = io.BytesIO()
    # sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
    # bytes_io.seek(0)

    start = time.time()
    if(lan=='en'):
        model = whisper.load_model("base.en") # Change this to your desired model
        transcribe = model.transcribe('sound.wav',fp16=False)
    elif(lan=='ga'):
        model = whisper.load_model("medium") # Change this to your desired model
        transcribe = model.transcribe('sound.wav',fp16=False, language='German')
    segments = transcribe['segments']
    end = time.time()
    print(end - start, "for transcribe")
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
    return send_file('output.zip', as_attachment=True)


@app.route("/delete", methods=["GET", "POST"])
def delete():
    if os.path.exists('output.zip'):
        os.remove('output.zip')
    if os.path.exists('sub.srt'):
        os.remove('sub.srt')
    if os.path.exists('sound.wav'):
        os.remove('sound.wav')
    print("Deleted")
    return "Deleted"

    
@app.route("/summarise", methods=["GET", "POST"])
def summarise():
    inp = request.get_json()
    lan = inp['lan']
    text = inp['text']
    if(text[0]=='"'):
        text=text[1:]
    if(text[-1]=='"'):
        text=text[:-1]
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('’', '')
    text = text.replace('‘', '')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("\r", '')
    text = text.replace("\n", '')
    text = text.replace('"\"', '')
    #whisper can summarise audio apparently
    try:
        start = time.time()
        words = text.split()
        print(words," words")
        word_count = len(words)
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        sentences = nltk.sent_tokenize(text)
        clean_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
            words = [wordnet_lemmatizer.lemmatize(word) for word in words]
            clean_sentences.append(' '.join(words))
        vectorizer = TfidfVectorizer()
        vectorized_sentences = vectorizer.fit_transform(clean_sentences)
        svd_model = TruncatedSVD(n_components=1)
        svd_model.fit(vectorized_sentences)
        sentence_scores = vectorized_sentences.dot(svd_model.components_.T)
        threshold = np.mean(sentence_scores)
        summary = ''

        num_sentences = 0
        for i, sentence in enumerate(sentences):
            if sentence_scores[i] > threshold and num_sentences < 4:
                summary += sentence + ' '
                num_sentences += 1
            elif num_sentences >= 4:
                break
        summary = summary.strip()
        summary_word = summary.split()
        summary_count = len(summary_word)
        summary_count=len(summary_word)
        end = time.time()
        print(end - start)
        print('LSA',summary)
        if(lan=='en'):
            tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
        elif(lan=='ga'):
            tts = TTS(model_name="tts_models/ga/cv/vits", progress_bar=False, gpu=False)
        else:
            return jsonify({"error": "Unknown voice model"}), 400
        audio=tts.tts(text=summary)
        tts.tts_to_file(summary, file_path="sound.wav")
        bytes_io = io.BytesIO()
        sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
        bytes_io.seek(0)
        audio_base64 = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
        return {'text': summary, 'audio': audio_base64}
        # if summary_count<=word_count//10:
        #     print(10)
        #     return  summary
        # else:
        #     print(15)
        #     x=len(summary) // 15
        #     summary=summary[:x+1]
        #     print(summary)
        #     summ = summary[:summary.rfind(".") + 1]
        #     print(summ)
        #     return  summ
    except Exception as e:
        print('elsa', e)
        return "error"
        
 # text = summary
            # print(summary)
            # tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
            # audio=tts.tts(text=text)
            # tts.tts_to_file(text, file_path="sound.wav")
            # bytes_io = io.BytesIO()
            # sf.write(bytes_io, audio, samplerate=22050, format='WAV', subtype='PCM_16')
            # bytes_io.seek(0)

if __name__ == "__main__":
    app.run(port=5000, debug=True)