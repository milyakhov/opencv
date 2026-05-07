from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from collections import Counter
import numpy as np

model = YOLO('yolov8n.pt')  # Быстрая модель

input_folder = "output_sliced_video1"
output_folder = "pz5_png_fixed"
output_data = "pz5_results.json"

Path(output_folder).mkdir(exist_ok=True)

# 🔧 ИСПРАВЛЕНИЕ: PNG + JPG
frames = sorted(Path(input_folder).glob("*.png")) + sorted(Path(input_folder).glob("*.jpg"))
print(f"Найдено PNG: {len(list(Path(input_folder).glob('*.png')))}")
print(f"Найдено JPG: {len(list(Path(input_folder).glob('*.jpg')))}")
print(f"Всего кадров: {len(frames)}")

results = {}
found_frames = 0

for i, frame_path in enumerate(frames[:50]):  # Первые 50
    # PNG требует флага IMREAD_UNCHANGED
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Не удалось загрузить: {frame_path.name}")
        continue
    

    # ПРОСТОЙ ИНФЕРЕНС
    res = model(img, conf=0.1, verbose=False)[0]
    
    frame_objects = []
    if res.boxes is not None:
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf > 0.1:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                
                frame_objects.append({
                    "class": name,
                    "confidence": round(conf, 3),
                    "bbox": [int(c) for c in xyxy]
                })
                
                # Рамка
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{name} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if frame_objects:
        results[str(frame_path.name)] = frame_objects
        out_path = Path(output_folder) / frame_path.name
        cv2.imwrite(str(out_path), img)
        found_frames += 1

# СОХРАНЕНИЕ
with open(output_data, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"ГОТОВО! Найдено объектов: {found_frames}/{len(frames)}")
print(f"Результы: {output_folder}/")