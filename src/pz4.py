import os
import json
import pywhisper
from moviepy.editor import VideoFileClip

video_path = "test.mp4"
audio_path = "temp_audio.wav"
output_txt = "pz4_transcription.txt"
output_json = "pz4_transcription.json"

if not os.path.exists(audio_path):
    print("Извлечение аудиодорожки...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le", verbose=False, logger=None)
    video.close()
    print(f"Аудио сохранено в {audio_path}")

print("Загрузка модели Whisper...")
model = pywhisper.load_model("small")

print("Распознавание речи...")
result = model.transcribe(audio_path, language="ru", verbose=False)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Готово.")
print(result["text"])