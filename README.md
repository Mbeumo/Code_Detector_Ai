# Code Detector System

This project is a Django-based system that processes video files to detect code-related frames and audio segments. It uses machine learning models and OCR to classify video content.

---

## Features

- Upload a video file for processing.
- Detect frames containing code-related content.
- Extract audio segments with code-related keywords.
- View results, including classified frames and audio segments.

---

## Requirements

Before using the system, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (e.g., `venv` or `virtualenv`)
- **FFmpeg** (required for video/audio processing - see installation guide below)

### Internet Requirements

- **Required**: The first time you run the system, an internet connection is needed to download pretrained models (e.g., MobileNetV2 and Whisper). After the initial setup, the system can work offline.
- **Not Required**: For OCR (Pytesseract) and video/audio processing (MoviePy), no internet connection is needed.

---

## Installation Guide

### 1. Install

#### Windows

**Option 1: Using Chocolatey (Recommended)**
    ```bash
    choco install ffmpeg
**Option 2: Manual Installation**

--**Download the "essentials" ZIP from**

    ```bash
    https://www.gyan.dev/ffmpeg/builds/
    ```
--**Extract to here**

    ```bash 
    C:\ffmpeg
    ``
--**Add to PATH: Press**
    ```bash
    Win + S
    ```
    → "Environment Variables"

--**Edit System Variables → Path → New:**
    ```bash
    C:\ffmpeg\bin
    ``
--**Restart your terminal/IDE**

#### macOS

    ```bash
    brew install ffmpeg`
    ```

#### Linux (Debian/Ubuntu)

    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

### 2. Clone the Repository

    ```bash 
    git clone <repository-url>
    cd <repository-folder>
    ```

### 3. Set Up Virtual Environment

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux

    python3 -m venv venv
    source venv/bin/activate
    ```

### 4. Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```

### 5. Set Up Django

    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

## Usage Guide

### 1. Navigate to the Upload Page

    ```bash
    <http://127.0.0.1:8000/upload/>
    ```

### 2. Upload a Video File`

-**Use the upload form to select a video file**

-**Click "Upload" to start processing**

### 3. Wait for Processing

--**Frames are analyzed using OCR and ML models**

-**Audio is transcribed and scanned for code keywords**

-**Errors appear for unsupported formats**

### 4. View Results
    ```bash
    <http://127.0.0.1:8000/result/>
    ```

-**Code Frames: Visual content with code**

--**Non-Code Frames: Other visual content**

--**Code Audio: Segments mentioning code terms**

## File Structure

--**views.py: Handles upload processing**

--**urls.py: Manages URL routing**

--**video_processor.py: Core processing logic**

--**templates/: HTML pages for UI**

## Troubleshooting

## **FileNotFoundError: [WinError 2] The system cannot find the file specified:**

--**Verify installation with ffmpeg -version**

-**Ensure PATH includes FFmpeg binaries**

## **Missing Tesseract:**

--**Install Tesseract OCR: Windows | macOS | Linux**

## **Model Download Issues:**

--**Ensure internet connection on first run**

--**Check firewall/proxy settings**

## License

--**MIT License**
