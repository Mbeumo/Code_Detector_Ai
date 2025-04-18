import os
import cv2
import numpy as np
import shutil
from pytesseract import image_to_string
from PIL import Image
import torch
import whisper

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

FRAME_DIR = "media/frames"
DATASET_DIR = "media/dataset"
AUDIO_PATH = "media/audio.wav"
CODE_LABELS = ['screen', 'monitor', 'desktop_computer', 'laptop']
CODE_KEYWORDS = ["int", "string", "class", "public", "void", "def", "print", "return", "function", ";", "if", "else"]

model = MobileNetV2(weights="imagenet", include_top=True)

def setup_dirs():
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/code", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/non_code", exist_ok=True)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(FRAME_DIR, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_id += 1
    cap.release()

def is_code_frame(img):
    image = cv2.resize(img, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    decoded = decode_predictions(preds, top=3)[0]
    for (_, label, _) in decoded:
        if label in CODE_LABELS:
            return True
    return False

def detect_code_via_ocr(image_path):
    try:
        text = image_to_string(Image.open(image_path))
        for keyword in CODE_KEYWORDS:
            if keyword.lower() in text.lower():
                return True
    except:
        pass
    return False

def classify_frames():
    for filename in os.listdir(FRAME_DIR):
        path = os.path.join(FRAME_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        if is_code_frame(img) or detect_code_via_ocr(path):
            shutil.copy(path, os.path.join(DATASET_DIR, "code", filename))
        else:
            shutil.copy(path, os.path.join(DATASET_DIR, "non_code", filename))

def extract_audio(video_path):
    import moviepy.editor as mp
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(AUDIO_PATH)

def classify_audio():
    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_PATH)
    code_lines = []
    for segment in result['segments']:
        text = segment['text'].lower()
        if any(kw in text for kw in CODE_KEYWORDS):
            code_lines.append(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
    return code_lines

def classify_video(video_path):
    setup_dirs()
    extract_frames(video_path)
    classify_frames()
    extract_audio(video_path)
    code_audio = classify_audio()
    return {
        "code_frames": os.listdir(os.path.join(DATASET_DIR, "code")),
        "non_code_frames": os.listdir(os.path.join(DATASET_DIR, "non_code")),
        "code_audio_segments": code_audio
    }
