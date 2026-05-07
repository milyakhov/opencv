import cv2
import easyocr
import time
import os

# Инициализация EasyOCR один раз (для 'eng')
reader = easyocr.Reader(['en'], gpu=True)  # gpu=False для CPU

cap = cv2.VideoCapture('video-text.mp4')
frame_count = 0

c = list(r"“'(){}[]?/\|-_*+!@#№$%^&=;:1234567890~»«.,'""¥—”¢”°><®")

while True:
    success, img = cap.read()
    if not success: 
        break
    
    frame_count += 1
    
    # OCR только каждые 30 кадров (1 сек при 30fps)
    if frame_count % 30 == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # EasyOCR на grayscale
        results = reader.readtext(gray)
        
        # Собираем весь текст из результатов
        text = ' '.join([result[1] for result in results if len(result[1]) > 0])
        
        # Удаляем символы
        for char in c:
            text = text.replace(char, "")

        if len(text) > 1:
            print(f"Frame {frame_count}: {text.strip()}")
    
    time.sleep(0.05)  # 50мс пауза
    #if cv2.waitKey(1) & 0xFF == ord('q'): 
        #break

cap.release()
#cv2.destroyAllWindows()