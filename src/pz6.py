# pz6_resnet_sliding_window.py
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import json
from PIL import Image
import numpy as np
from collections import Counter

# Загрузка ResNet50 (ImageNet 1000 классов)
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

input_folder = "output_sliced_video1"  # Твоя папка с PNG
output_data = "pz6_resnet_objects.json"
output_folder = "pz6_resnet_windows"

Path(output_folder).mkdir(exist_ok=True)

# SLIDING WINDOW параметры
WINDOW_SIZE = (224, 224)  # ResNet input size
STRIDE = 32  # Шаг окна (меньше = больше окон)
CONFIDENCE_THRESHOLD = 0.7  # Только уверенные детекции

print("ResNet50 + SLIDING WINDOW детекция объектов")
print(f"Окно: {WINDOW_SIZE}, шаг: {STRIDE}")

# PNG + JPG
frames = sorted(Path(input_folder).glob("*.png")) + sorted(Path(input_folder).glob("*.jpg"))
print(f"Кадров: {len(frames)}")

results = {}
total_detections = 0

for frame_idx, frame_path in enumerate(frames[:100]):  # Первые 100 для скорости
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        continue
    
    height, width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    frame_detections = []
    
    # SLIDING WINDOW
    for y in range(0, height - WINDOW_SIZE[1], STRIDE):
        for x in range(0, width - WINDOW_SIZE[0], STRIDE):
            # Вырезаем окно
            window = img_rgb[y:y+WINDOW_SIZE[1], x:x+WINDOW_SIZE[0]]
            if window.shape[:2] != WINDOW_SIZE:
                continue
                
            pil_window = Image.fromarray(window)
            batch = preprocess(pil_window).unsqueeze(0)
            
            with torch.no_grad():
                prediction = model(batch).squeeze(0).softmax(0)
                class_id = prediction.argmax().item()
                score = prediction[class_id].item()
                
                if score > CONFIDENCE_THRESHOLD:
                    category = weights.meta["categories"][class_id]
                    
                    # Координаты окна
                    bbox = [x, y, x+WINDOW_SIZE[0], y+WINDOW_SIZE[1]]
                    
                    frame_detections.append({
                        "class": category,
                        "confidence": round(score, 3),
                        "window_bbox": bbox,
                        "position": "center"
                    })
                    
                    total_detections += 1
                    
                    # Рисуем окно на оригинале
                    cv2.rectangle(img, (x, y), (x+WINDOW_SIZE[0], y+WINDOW_SIZE[1]), 
                                (255, 0, 0), 1)
                    cv2.putText(img, f"{category[:10]}:{score:.2f}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    if frame_detections:
        results[str(frame_path.name)] = frame_detections
        
        # Сохраняем кадр с окнами
        out_path = Path(output_folder) / frame_path.name
        cv2.imwrite(str(out_path), img)
    
    if frame_idx % 20 == 0:
        print(f"Обработано: {frame_idx+1}/{min(100, len(frames))}")

# СТАТИСТИКА
stats = {
    "total_frames": len(frames),
    "processed_frames": len([f for f in frames if len(f.name) > 0][:100]),
    "frames_with_detections": len(results),
    "total_detections": total_detections,
    "top_classes": Counter([d["class"] for dets in results.values() for d in dets]).most_common(10),
    "settings": {
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }
}

# СОХРАНЕНИЕ
final_results = {"detections": results, "statistics": stats}
with open(output_data, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print("\nResNet SLIDING WINDOW завершён!")
print(f"Найдено окон: {total_detections}")
print("Топ классы:")
for cls, count in stats["top_classes"]:
    print(f"   {cls}: {count}")
print(f"Кадры с окнами: {output_folder}/")