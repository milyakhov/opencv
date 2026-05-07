import cv2
import os
import pytesseract
from pytesseract import Output

name = "371412.png"
input_path = os.path.join("dataset", name)
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

img = cv2.imread(input_path)
if img is None:
    print("Ошибка загрузки изображения")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

inverted = cv2.bitwise_not(gray)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

cv2.imwrite(os.path.join(output_folder, f"binary{name[0:6]}.png"), binary)

print("Бинаризация завершена. Результаты в папке:", output_folder)

custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789" 

data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)

for i in range(len(data['text'])):
    text = data['text'][i].strip()
    height = data['height'][i]
    
    if text and len(text) >= 6 and height > 50:
        print("НАЙДЕН НОМЕР:", text[:6])
        found_number = text
        break
