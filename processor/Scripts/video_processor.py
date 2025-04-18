import os
import cv2
import numpy as np
import shutil
from pytesseract import image_to_string
from PIL import Image
import torch
import whisper
from datetime import timedelta, datetime
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# Create a unique prefix based on timestamp or filename
prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
FRAME_DIR = "media/frames"
DATASET_DIR = "media/dataset"
AUDIO_PATH = f"media/audio{prefix}.wav"
CODE_LABELS = ['screen', 'monitor', 'desktop_computer', 'laptop']
CODE_KEYWORDS = [
    "int", "string", "float", "double", "char", "boolean",
    "class", "struct", "interface", "public", "private", "protected", "static",
    "void", "def", "function", "lambda", "return", "yield", "async", "await",
    "print", "echo", "console.log", "System.out.println",
    "if", "else", "elif", "switch", "case", "break", "continue", "default",
    "for", "while", "do", "foreach", "range", "map",
    "import", "from", "using", "include", "#define",
    "try", "catch", "finally", "throw", "except",
    "new", "delete", "malloc", "free",
    "true", "false", "None", "null",
    "==", "!=", "<", ">", "<=", ">=", "&&", "||",
    "=", "+", "-", "*", "/", "%", "**",
    ";", "{", "}", "(", ")", "[", "]"
]

#can change this model to a custom model created 
#but no tguarnateed of the accuracy as not enougth data is provided
model = MobileNetV2(weights="imagenet", include_top=True)

def setup_dirs():
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/code", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/non_code", exist_ok=True)

def extract_frames(video_path, prefix):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"{prefix}_frame_{frame_id:04d}.jpg"
        frame_path = os.path.join(FRAME_DIR, frame_filename)
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

def classify_frames(prefix, fps):
    frame_names = sorted([f for f in os.listdir(FRAME_DIR) if f.startswith(prefix)])
    code_timestamps = []

    for i, frame_name in enumerate(frame_names):
        frame_path = os.path.join(FRAME_DIR, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        if is_code_frame(img):
            dest_path = os.path.join(DATASET_DIR, "code", frame_name)
            if not os.path.exists(dest_path):
                shutil.copy(frame_path, dest_path)
            seconds = i // fps
            time_str = str(timedelta(seconds=seconds))
            code_timestamps.append((frame_name, time_str))
        else:
            dest_path = os.path.join(DATASET_DIR, "non_code", frame_name)
            if not os.path.exists(dest_path):
                shutil.copy(frame_path, dest_path)

    return code_timestamps

def extract_audio(video_path):
    import moviepy.video.io.VideoFileClip as mp
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(AUDIO_PATH)

def classify_audio():
    try:
        # Load the Whisper model
        model = whisper.load_model("base")
        print("Whisper model loaded successfully.")

        # Transcribe the audio file
        result = model.transcribe(AUDIO_PATH)
        print("Audio transcription completed.")

        code_lines = []
        for segment in result.get('segments', []):
            text = segment['text'].lower()
            if any(kw in text for kw in CODE_KEYWORDS):
                print(f"Code-related audio detected: {segment['text']}")
                code_lines.append({
                    "start": str(timedelta(seconds=segment['start'])),
                    "end": str(timedelta(seconds=segment['end'])),
                    "text": segment['text']
                })

        print(f"Total code-related audio segments detected: {len(code_lines)}")
        return code_lines

    except FileNotFoundError:
        print(f"Error: Audio file not found at {AUDIO_PATH}.")
        return []

    except Exception as e:
        print(f"Error during audio classification: {e}")
        return []

def classify_video(video_path):
    try:
        setup_dirs()

        # Extract frames using this prefix
        extract_frames(video_path, prefix)
        print("Frames extracted successfully.")

        # Get FPS
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        print(f"Video FPS: {fps}")

        # Classify frames
        code_frame_timestamps = classify_frames(prefix, fps)
        print(f"Total code frames detected: {len(code_frame_timestamps)}")

        # Extract and classify audio
        extract_audio(video_path)
        print("Audio extracted successfully.")
        code_audio = classify_audio()

        # Write results to summary file
        result_txt_path = os.path.join("media", f"code_summary_{prefix}.txt")
        with open(result_txt_path, "w", encoding="utf-8") as f:
            f.write("=== CODE FRAMES DETECTED (with Timestamps) ===\n")
            for frame_name, timestamp in code_frame_timestamps:
                f.write(f"{timestamp}: {frame_name}\n")

            f.write("\n=== CODE AUDIO SEGMENTS ===\n")
            for line in code_audio:
                f.write(f"{line['start']} - {line['end']}: {line['text']}\n")

        print(f"Summary file written successfully: {result_txt_path}")
        return {
            "code_frames": [item[0] for item in code_frame_timestamps],
            "non_code_frames": os.listdir(os.path.join(DATASET_DIR, "non_code")),
            "code_audio_segments": code_audio,
            "summary_txt": result_txt_path
        }

    except Exception as e:
        print(f"Error during video classification: {e}")
        return {
            "code_frames": [],
            "non_code_frames": [],
            "code_audio_segments": [],
            "summary_txt": None
        }