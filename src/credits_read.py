import cv2
import easyocr
import os
import re
import yt_dlp
from thefuzz import fuzz

# --- НАСТРОЙКИ ---
VIDEO_SOURCE = "https://rutube.ru/video/cb58fcad6be06c73fa0b16fe527cfb3a/" # Вставь ссылку
OUTPUT_FILE = "titles/results.txt"
OCR_LANG = ['ru']
GPU_ENABLED = True
FRAME_SKIP = 30
CONFIDENCE_THRESHOLD = 0.4

# Порог схожести для частичного совпадения
# partial_ratio более чувствителен к вложенности строк
PARTIAL_SIMILARITY_THRESHOLD = 85 
# -----------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def get_direct_url(source):
    if source.startswith(('http://', 'https://')):
        print("Извлечение прямой ссылки...")
        ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'quiet': True, 'no_warnings': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                return info['url']
        except Exception as e:
            print(f"Ошибка yt-dlp: {e}")
            return None
    return source

print("Загрузка моделей OCR...")
reader = easyocr.Reader(OCR_LANG, gpu=GPU_ENABLED)

print("Подключение к видео...")
direct_url = get_direct_url(VIDEO_SOURCE)
if not direct_url:
    exit()

cap = cv2.VideoCapture(direct_url)
if not cap.isOpened():
    print("Ошибка открытия видео.")
    exit()

def clean_raw_text(text):
    """Базовая очистка от явного мусора"""
    cleaned = re.sub(r'[^\w\s\-–—]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def is_duplicate_advanced(new_text, history):
    """
    Продвинутая проверка на дубликаты.
    1. Проверяет полное нечеткое совпадение (для случаев, когда текст почти идентичен).
    2. Проверяет частичное совпадение (для случаев, когда текст является продолжением или частью предыдущего).
    """
    if len(new_text) < 3: 
        return True
        
    new_lower = new_text.lower()
    
    for prev_text in history:
        prev_lower = prev_text.lower()
        
        # 1. Частичное совпадение (Самое важное для твоей проблемы)
        # Проверяет, насколько лучшая совпадающая подстрока похожа на shorter string
        partial_score = fuzz.partial_ratio(new_lower, prev_lower)
        
        if partial_score >= PARTIAL_SIMILARITY_THRESHOLD:
            return True
            
        # 2. Полное совпадение (на случай, если длины равны, но есть опечатки)
        # Иногда partial_ratio может дать высокий балл даже для разных коротких слов,
        # поэтому для очень коротких строк лучше использовать token_sort_ratio или ratio
        if len(new_text) > 5 and len(prev_text) > 5:
             full_score = fuzz.ratio(new_lower, prev_lower)
             if full_score >= 90:
                 return True
                
    return False

recognized_texts = []
history_buffer = [] 
frame_count = 0

print("Обработка началась...")
try:
    while True:
        success, img = cap.read()
        if not success: break
        
        frame_count += 1
        if frame_count % FRAME_SKIP != 0: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=1, paragraph=False)
        
        # Сортировка сверху вниз
        sorted_results = sorted(results, key=lambda x: x[0][0][1])
        
        current_parts = []
        for (bbox, text, confidence) in sorted_results:
            if confidence > CONFIDENCE_THRESHOLD:
                cleaned = clean_raw_text(text)
                if len(cleaned) > 1:
                    current_parts.append(cleaned)
        
        if not current_parts: continue
        
        full_text_frame = " ".join(current_parts)
        
        # Используем новую функцию проверки
        if not is_duplicate_advanced(full_text_frame, history_buffer):
            recognized_texts.append(full_text_frame)
            history_buffer.append(full_text_frame)
            
            # Храним чуть больше истории, чтобы ловить длинные переходы
            if len(history_buffer) > 8: 
                history_buffer.pop(0)
                
            print(f"Frame {frame_count}: {full_text_frame}")

finally:
    cap.release()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(recognized_texts))
    print(f"\nГотово. Сохранено {len(recognized_texts)} строк в {OUTPUT_FILE}")