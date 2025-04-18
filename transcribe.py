import whisper

model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\Briand Mbeumo\OneDrive\Documents\SE3A\Introduction to AI\code_detector\media\audio20250418_194814.wav")
print(result["text"])