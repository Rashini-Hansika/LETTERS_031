from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List
import tensorflow as tf
import easyocr
from autocorrect import Speller
import cv2
from pydub import AudioSegment
import speech_recognition as sr
import os
from firebase_admin import credentials, firestore, initialize_app

#from fastapi import FastAPI, UploadFile
from transformers import pipeline
#from pydantic import BaseModel

import tempfile

import pytesseract


app = FastAPI()
# DB Configurations
cred = credentials.Certificate("./configs/serviceAccountKey.json")
firebase_app = initialize_app(cred)
db = firestore.client()

origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("keras_modelv2.h5")
CLASS_NAMES =  [' ']
target_width = 224
target_height = 224
spell = Speller('en')

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def convert_audio_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)
    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")
    return wav_path

#Audio To English Text
@app.post("/audio-to-text-en")
async def audio_text(files: List[UploadFile]):
    extracted_text = ""

    for file in files:
        if not file.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only WAV audio files are supported.")

        # Save 
        with open("temp_audio.wav", "wb") as audio_file:
            audio_file.write(file.file.read())

        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

            try:
                # Recognize the English audio using Google Web Speech API
                text = recognizer.recognize_google(audio, language="en-US")
                extracted_text += text + " "
            except sr.UnknownValueError:
                extracted_text += " [Unable to recognize] "
            except sr.RequestError:
                extracted_text += " [Recognition request failed] "

        
        os.remove("temp_audio.wav")

    return {"predictions": extracted_text}

#Speed clculation
def calculate_audio_speed(audio_path):
    # Load the audio file
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)

    # Calculate the speed (duration) of the audio in seconds
    duration = len(audio.frame_data) / audio.sample_rate

    return duration

@app.post("/compare-audio-speed")
async def compare_audio_speed(existing_audio: UploadFile, new_audio: UploadFile):
    # Save the uploaded files as temporary WAV files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as existing_audio_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as new_audio_file:
        existing_audio_data = existing_audio.file.read()
        new_audio_data = new_audio.file.read()
        
        existing_audio_file.write(existing_audio_data)
        new_audio_file.write(new_audio_data)

    # Calculate the speed of both audio files
    existing_audio_speed = calculate_audio_speed(existing_audio_file.name)
    new_audio_speed = calculate_audio_speed(new_audio_file.name)

    os.remove(existing_audio_file.name)
    os.remove(new_audio_file.name)

    # Compare the speeds
    if existing_audio_speed == new_audio_speed:
        speed_comparison = "Both audios have the same speed."
    elif existing_audio_speed < new_audio_speed:
        speed_comparison = "The new audio is faster than the existing audio."
    else:
        speed_comparison = "The new audio is slower than the existing audio."

    return {"speed_comparison": speed_comparison}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8000")