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

def classify_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_names = sorted(os.listdir(FRAME_DIR))
    code_timestamps = []

    for i, frame_name in enumerate(frame_names):
        frame_path = os.path.join(FRAME_DIR, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        if is_code_frame(img):
            shutil.copy(frame_path, f"{DATASET_DIR}/code/{frame_name}")
            seconds = i // fps
            time_str = str(timedelta(seconds=seconds))
            code_timestamps.append((frame_name, time_str))
        else:
            shutil.copy(frame_path, f"{DATASET_DIR}/non_code/{frame_name}")

    cap.release()
    return code_timestamps

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
    code_frame_timestamps = classify_frames(video_path)
    extract_audio(video_path)
    code_audio = classify_audio()

    result_txt_path = os.path.join("media", "code_summary.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("=== CODE FRAMES DETECTED (with Timestamps) ===\n")
        for frame_name, timestamp in code_frame_timestamps:
            f.write(f"{timestamp}: {frame_name}\n")

        f.write("\n=== CODE AUDIO SEGMENTS ===\n")
        for line in code_audio:
            f.write(f"{line}\n")

    return {
        "code_frames": [item[0] for item in code_frame_timestamps],
        "non_code_frames": os.listdir(os.path.join(DATASET_DIR, "non_code")),
        "code_audio_segments": code_audio,
        "summary_txt": result_txt_path
    }

