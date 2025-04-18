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

### Internet Requirements
- **Required**: The first time you run the system, an internet connection is needed to download pretrained models (e.g., MobileNetV2 and Whisper). After the initial setup, the system can work offline.
- **Not Required**: For OCR (Pytesseract) and video/audio processing (MoviePy), no internet connection is needed.

---

## Installation Guide

### 1. Clone the Repository
    ```bash
    git clone <repository-url>
    cd <repository-folder>
### 2. Set Up a Virtual Environment
Create and activate a virtual environment:
    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

### 3. Install Dependencies
Install the required Python packages using requirements.txt:
    ```bash
    pip install -r requirements.txt

### 4.Set Up Django
Run the following commands to set up the Django project:
    ```bash
    # Apply migrations
    python manage.py migrate

# Start the development server
python manage.py runserver

## Usage Guide

### 1. Navigate to the Upload Page
Open your browser and go to: http://127.0.0.1:8000/upload/
### 2. Upload a Video File
Use the file upload form to select a video file.
Click the "Upload" button to start processing.
### 3. Wait for Processing
The system will process the video to classify frames and extract audio segments.
If there are any errors (e.g., unsupported file format), an error message will be displayed.
### 4. View Results
After processing, you will be redirected to the results page: http://127.0.0.1:8000/result/
The results page will display:
Code Frames: Frames containing code-related content.
Non-Code Frames: Frames without code-related content.
Code-related Audio Segments: Audio segments with code-related keywords.
File Structure
views.py: Contains the logic for handling video uploads and processing.
urls.py: Defines the URL routes for the system.
video_processor.py: Handles video frame extraction, classification, and audio processing.
templates/: Contains the HTML templates for the upload and result pages.
Notes
Ensure the media directory exists in the project root for storing uploaded files, frames, and results.
The system uses pretrained models (MobileNetV2 and Whisper). These models will be downloaded automatically if not already cached.
Troubleshooting
Error: Missing Tesseract: Ensure Tesseract OCR is installed on your system. Installation Guide.
Error: Missing Models: Ensure you have an internet connection for the first run to download pretrained models.
License
This project is licensed under the MIT License.
```bash 

---

### Explanation:
1. **Requirements**: Lists Python version, dependencies, and internet requirements.
2. **Installation Guide**: Explains how to set up the project, including virtual environment and dependencies.
3. **Usage Guide**: Walks the user through navigating the system, uploading a video, and viewing results.
4. **Troubleshooting**: Provides solutions for common issues like missing Tesseract or models.
5. **File Structure**: Gives an overview of the key files in the project.
---

### Explanation:
1. **Requirements**: Lists Python version, dependencies, and internet requirements.
2. **Installation Guide**: Explains how to set up the project, including virtual environment and dependencies.
3. **Usage Guide**: Walks the user through navigating the system, uploading a video, and viewing results.
4. **Troubleshooting**: Provides solutions for common issues like missing Tesseract or models.
5. **File Structure**: Gives an overview of the key files in the project.