import re
from thefuzz import fuzz

INPUT_FILE = "titles/results.txt"
OUTPUT_FILE = "titles/results_cleaned.txt"

# Порог схожести. Для частичных совпадений можно снизить до 75-80,
# чтобы ловить даже искаженные части.
SIMILARITY_THRESHOLD = 70 

def clean_text_for_compare(text):
    """Приводим текст к виду, удобному для сравнения"""
    text = text.lower()
    # Убираем все кроме букв и пробелов (убираем пунктуацию, чтобы она не мешала)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_duplicates_aggressive(lines):
    cleaned_lines = []
    
    print(f"Начало фильтрации {len(lines)} строк...")
    
    for i, current_line in enumerate(lines):
        if not current_line.strip():
            continue
            
        current_clean = clean_text_for_compare(current_line)
        is_dup = False
        
        # Сравниваем с уже сохраненными строками
        # Мы будем проверять два условия:
        # 1. Текущая строка является частью одной из сохраненных
        # 2. Одна из сохраненных строк является частью текущей
        
        to_remove_indices = [] # Индексы строк в cleaned_lines, которые нужно удалить (если текущая их перекрывает)

        for idx, prev_line in enumerate(cleaned_lines):
            prev_clean = clean_text_for_compare(prev_line)
            
            # Быстрая проверка: если строки идентичны после очистки
            if current_clean == prev_clean:
                is_dup = True
                break
            
            # Проверка: является ли текущая строка ПОДСТРОКОЙ предыдущей?
            # Если ДА, значит текущая - это просто обрезанный кусок, он нам не нужен.
            if current_clean in prev_clean:
                # Но нужно убедиться, что они похожи по смыслу (чтобы "иван" не удалилось из "иванов")
                # Используем partial_ratio для надежности
                score = fuzz.partial_ratio(current_clean, prev_clean)
                if score > 90: 
                    is_dup = True
                    break

            # Проверка: является ли предыдущая строка ПОДСТРОКОЙ текущей?
            # Если ДА, значит предыдущая была обрезком, а текущая - полная версия.
            # Удаляем предыдущую из списка, а текущую добавляем.
            if prev_clean in current_clean:
                score = fuzz.partial_ratio(prev_clean, current_clean)
                if score > 90:
                    to_remove_indices.append(idx)
                    
            # Если ни одна не входит в другую полностью, проверяем сильное частичное перекрытие
            # Это нужно для случаев с опечатками (режшссер vs режиссер)
            if not (current_clean in prev_clean or prev_clean in current_clean):
                 score = fuzz.partial_ratio(current_clean, prev_clean)
                 if score >= SIMILARITY_THRESHOLD:
                     # Если相似度 высокая, смотрим на длину. 
                     # Более длинная строка обычно информативнее.
                     if len(current_clean) <= len(prev_clean):
                         is_dup = True # Текущая хуже или равна, отбрасываем
                         break
                     else:
                         to_remove_indices.append(idx) # Текущая лучше, удаляем старую

        if is_dup:
            continue # Пропускаем текущую строку
            
        # Если текущая строка лучше некоторых предыдущих, удаляем предыдущие
        # Идем с конца, чтобы не сбить индексы
        for idx in sorted(to_remove_indices, reverse=True):
            del cleaned_lines[idx]
            
        # Добавляем текущую строку
        cleaned_lines.append(current_line)
            
    return cleaned_lines

# --- Основной процесс ---
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        print("Файл пуст.")
    else:
        unique_lines = filter_duplicates_aggressive(lines)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for line in unique_lines:
                f.write(line + '\n')
                
        print(f"Готово!")
        print(f"Было строк: {len(lines)}")
        print(f"Стало строк: {len(unique_lines)}")
        print(f"Удалено дубликатов: {len(lines) - len(unique_lines)}")
        print(f"Результат сохранен в: {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"Файл {INPUT_FILE} не найден.")
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()