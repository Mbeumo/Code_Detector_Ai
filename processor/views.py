from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .scripts.video_processor import classify_video
import os

def save_uploaded_file(file):
    """Helper function to save uploaded file and return its full path."""
    fs = FileSystemStorage()
    file_path = fs.save(file.name, file)
    return os.path.join(fs.location, file_path), fs.url(file_path)

def upload_video(request):
    """Handle video upload and classification."""
    context = {}
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        
        # Save the uploaded video
        try:
            video_full_path, video_url = save_uploaded_file(video)
        except Exception as e:
            context["error"] = f"Error saving file: {str(e)}"
            return render(request, "upload.html", context)

        # Process the video
        try:
            result = classify_video(video_full_path)
        except Exception as e:
            context["error"] = f"Error processing video: {str(e)}"
            return render(request, "upload.html", context)

        # Store results in session
        request.session['summary_path'] = result.get("summary_txt", "")
        request.session['code_frames'] = result.get("code_frames", "")
        request.session['non_code_frames'] = result.get("non_code_frames", "")

        context.update({
            "video_url": video_url,
            "code_frames": result.get("code_frames", []),
            "non_code_frames": result.get("non_code_frames", []),
            "code_audio_segments": result.get("code_audio_segments", []),
        })

        return render(request, "result.html", context)
        #return redirect('result_page')

    return render(request, "upload.html", context)

def result_page(request):
    """Display the result page with summary and frames."""
    summary_path = request.session.get('summary_path')
    frames = {
        "code_frames": request.session.get("code_frames", []),
        "non_code_frames": request.session.get("non_code_frames", []),
        "code_audio_segments": request.session.get("code_audio_segments", []),
        "summary_path": summary_path,
    }
    if summary_path:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_content = f.read()
            frames["summary"] = summary_content
        except FileNotFoundError:
            frames["error"] = "Summary file not found."
        except Exception as e:
            frames["error"] = f"Error reading summary file: {str(e)}"

    print(frames)

    return render(request, "result.html", {"frames": frames})