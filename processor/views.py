from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .scripts.video_processor import classify_video
import os

def upload_video(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage()
        video_path = fs.save(video.name, video)
        video_full_path = os.path.join(fs.location, video_path)
        
        result = classify_video(video_full_path)

        context["video_url"] = fs.url(video_path)
        context["code_frames"] = result["code_frames"]
        context["non_code_frames"] = result["non_code_frames"]
        context["code_audio_segments"] = result["code_audio_segments"]

    return render(request, "upload.html", context)
