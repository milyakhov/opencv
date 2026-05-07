import cv2
import os
import time

cap = cv2.VideoCapture('video.mp4')
frame_count = 0

output_folder = "output_sliced_video"
os.makedirs(output_folder, exist_ok=True)

#c = list(r"“'(){}[]?/\|-_*+!@#№$%^&=;:1234567890~»«.,'""¥—”¢”°><®")
k = 0
#параметр частоты нарезки кадров
frame_per_sec = 30

while True:
    success, img = cap.read()
    if not success: break
    
    frame_count += 1
    
    if frame_count % frame_per_sec == 0:
        k += 1
        cv2.imwrite(os.path.join(output_folder, f"frame{k}.png"), img)

    time.sleep(0.05)  # 50мс пауза
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()