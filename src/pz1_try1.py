import cv2
import os
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

input_path = os.path.join("dataset", "371412.png")
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

img = cv2.imread(input_path)
if img is None:
    print("Ошибка загрузки изображения")
    cv2.destroyAllWindows()
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

median = cv2.medianBlur(gray, 3)
bilat = cv2.bilateralFilter(median, 1, 10, 10)

"""cv2.imshow("Result", bilat)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

binary = cv2.threshold(bilat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cv2.imwrite(os.path.join(output_folder, "binary.png"), binary)

print("Бинаризация завершена, результаты в папке:", output_folder)

#конфиг для pytesseract распознование цифр 
custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789"

text = pytesseract.image_to_string(binary, config=custom_config)

# убираем пробелы и переносы строк
text = text.replace(" ", "").replace("\n", "")

found_number = None

# можно искать все возможные 6-значные подстроки
matches = re.findall(r"\d{6}", text)
if matches:
    found_number = matches[0]
    print("Найден номер:", found_number)
else:
    print("6-значный номер не найден")