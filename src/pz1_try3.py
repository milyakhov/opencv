import cv2
import os
import pytesseract
from pytesseract import Output
import numpy as np

name = "371414.png"
input_path = os.path.join("dataset", name)
output_folder = "output_cleaned_v3"

os.makedirs(output_folder, exist_ok=True)

img = cv2.imread(input_path)
if img is None:
    print("Ошибка загрузки изображения")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Твоя предобработка
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
binary = cv2.adaptiveThreshold(
    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Инверсия
binary_inv = cv2.bitwise_not(binary)

# Утолщаем сильнее
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_inv_thick = cv2.dilate(binary_inv, kernel_dilate, iterations=2)  # iterations=2

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv_thick, connectivity=8)

# Параметры фильтра
min_area = 150  # уменьшил для "1"
min_height = 50  # высота компонента
aspect_min = 0.2  # ширина/высота для узких цифр как "1"
aspect_max = 1.0  # для квадратных как "0"

print("Площади и формы компонентов (отсеем area<", min_area, ", height<", min_height, ", aspect not in [", aspect_min, ",", aspect_max, "]):")
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    aspect = w / h if h > 0 else 0
    print(f"Компонент {i}: area={area}, width={w}, height={h}, aspect={aspect:.2f}")

# Чистая маска с фильтром
cleaned = np.zeros_like(binary_inv_thick)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    aspect = w / h if h > 0 else 0
    if area >= min_area and h >= min_height and aspect_min <= aspect <= aspect_max:
        cleaned[labels == i] = 255

# Закрытие для заполнения дыр
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # чуть больше
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

cv2.imwrite(os.path.join(output_folder, f"thick_inv_{name[:-4]}.png"), binary_inv_thick)
cv2.imwrite(os.path.join(output_folder, f"cleaned_inv_{name[:-4]}.png"), cleaned)

print("Сохранены thick_inv_*.png и cleaned_inv_*.png в", output_folder)

# Tesseract с psm 11
custom_config = r'--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'

data = pytesseract.image_to_data(cleaned, config=custom_config, output_type=Output.DICT)

found = False
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    height = data['height'][i]
    conf = data['conf'][i]
    if text.isdigit() and len(text) >= 6 and height > 45 and conf > 10:  # ещё снизил conf
        print("НАЙДЕН НОМЕР:", text, f"(conf={conf:.1f}, height={height})")
        found = True
        break

if not found:
    # Fallback на thick_inv
    data_fallback = pytesseract.image_to_data(binary_inv_thick, config=custom_config, output_type=Output.DICT)
    for i in range(len(data_fallback['text'])):
        text = data_fallback['text'][i].strip()
        height = data_fallback['height'][i]
        conf = data_fallback['conf'][i]
        if text.isdigit() and len(text) >= 6 and height > 45 and conf > 10:
            print("НАЙДЕН НОМЕР (fallback):", text, f"(conf={conf:.1f}, height={height})")
            found = True
            break
    if not found:
        print("Не найден. Поделись выводом print(компонентов) — подкрутим min_area/aspect по твоим площадям.")