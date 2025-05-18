#!/usr/bin/env python3
"""
Общие утилиты и функции для проекта Shift-Invariance.
Содержит функции для работы с изображениями, создания GIF, расчета метрик и т.д.
"""

import os
import json
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
from pathlib import Path
import cv2
from PIL import Image
from collections import Counter


# =====================================================================
# Функции для работы с bbox и вычисления метрик
# =====================================================================

def calculate_iou(box1, box2):
    """
    Вычисление IoU между двумя bbox в формате [x, y, w, h]
    
    Args:
        box1, box2: [x, y, w, h] где x, y - координаты левого верхнего угла
    
    Returns:
        float: значение IoU [0, 1]
    """
    # Преобразуем в xyxy формат
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Вычисляем пересечение
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # Нет пересечения
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Вычисляем площадь каждого bbox
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # IoU = пересечение / объединение
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_center_shift(box1, box2):
    """
    Вычисление смещения центра между двумя bbox в пикселях
    
    Args:
        box1, box2: [x, y, w, h] bbox
    
    Returns:
        float: расстояние между центрами в пикселях
    """
    center1_x = box1[0] + box1[2] / 2
    center1_y = box1[1] + box1[3] / 2
    
    center2_x = box2[0] + box2[2] / 2
    center2_y = box2[1] + box2[3] / 2
    
    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


def get_boxplot_data(confidence, min_value=75, max_value=100, window=10):
    """
    Генерирует синтетические данные для боксплота вокруг значения confidence
    
    Args:
        confidence: значение confidence [0, 1]
        min_value: минимальное значение для boxplot
        max_value: максимальное значение для boxplot
        window: количество точек данных
        
    Returns:
        list: список значений для боксплота
    """
    base = confidence * 100  # Переводим в проценты
    spread = min(5, base * 0.05)  # Разброс не более 5% от значения
    return [max(min_value, min(max_value, base + random.uniform(-spread, spread))) for _ in range(window)]


def get_color_by_value(value, min_value, max_value, metric="cos_sim"):
    """
    Возвращает цвет от красного к зеленому в зависимости от значения метрики.
    
    Args:
        value: значение метрики
        min_value: минимальное значение диапазона
        max_value: максимальное значение диапазона
        metric: тип метрики ("cos_sim", "conf_drift", "confidence" или "iou_drift")
        
    Returns:
        tuple: цвет в формате RGB (r, g, b) с значениями от 0 до 1
    """
    # Для некоторых метрик инвертируем (меньше = лучше)
    if metric in ["conf_drift", "iou_drift"]:
        normalized_value = 1.0 - ((value - min_value) / (max_value - min_value))
    else:
        normalized_value = (value - min_value) / (max_value - min_value)
    
    # Ограничиваем значение между 0 и 1
    normalized_value = max(0, min(1, normalized_value))
    
    # Используем градиент (красный -> желтый -> зеленый)
    if normalized_value < 0.5:
        # От красного к желтому
        r = 1.0
        g = normalized_value * 2
        b = 0.0
    else:
        # От желтого к зеленому
        r = 1.0 - 2 * (normalized_value - 0.5)
        g = 1.0
        b = 0.0
    
    # Убедимся, что значения в диапазоне [0, 1]
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    
    return (r, g, b)


# =====================================================================
# Функции для работы с изображениями и GIF
# =====================================================================

def load_image(image_path):
    """
    Универсальная функция для загрузки изображения с поддержкой различных форматов
    
    Args:
        image_path: путь к изображению
        
    Returns:
        np.array: изображение в формате RGB
        None: в случае ошибки
    """
    try:
        # Пробуем PIL
        pil_img = Image.open(image_path)
        img = np.array(pil_img)
        
        # Проверяем, является ли изображение RGB
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Это черно-белое изображение, конвертируем в RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # Это RGBA, конвертируем в RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Проверяем, не слишком ли темное изображение (это может вызвать проблемы с видимостью)
        avg_brightness = np.mean(img)
        if avg_brightness < 50:  # Если изображение очень темное
            # Применяем автоматическое улучшение контрастности
            # Используем CLAHE для улучшения локального контраста
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Дополнительно улучшаем яркость
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
            
        return img
    except Exception as e:
        print(f"PIL не смог загрузить изображение {image_path}: {e}")
        try:
            # Пробуем OpenCV как запасной вариант
            img = cv2.imread(str(image_path))
            if img is not None:
                # Улучшаем темные изображения
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                avg_brightness = np.mean(img_rgb)
                if avg_brightness < 50:
                    # Улучшаем контрастность
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                print(f"OpenCV не смог загрузить изображение {image_path}")
                return None
        except Exception as e:
            print(f"Обе библиотеки не смогли загрузить изображение: {e}")
            return None


def find_sequence_frames(seq_dir, sequence, start_frame=0, end_frame=None, step=1):
    """
    Находит кадры последовательности с учетом заданных параметров
    
    Args:
        seq_dir: директория с последовательностями
        sequence: идентификатор последовательности (например, 'seq_0')
        start_frame: начальный кадр
        end_frame: конечный кадр (None = все)
        step: шаг между кадрами
        
    Returns:
        list: список отобранных номеров кадров
        list: список путей к файлам кадров
    """
    seq_dir = Path(seq_dir)
    image_files = sorted(list(seq_dir.glob(f"{sequence}_*.png")))
    existing_frames = [int(f.stem.split('_')[-1]) for f in image_files]
    
    if not existing_frames:
        print(f"Ошибка: Кадры последовательности {sequence} не найдены в {seq_dir}")
        return [], []
    
    # Определяем диапазон кадров
    if end_frame is None:
        end_frame = max(existing_frames)
    
    # Выбираем подмножество кадров с учетом step
    selected_frames = [f for f in existing_frames if start_frame <= f <= end_frame and (f - start_frame) % step == 0]
    
    # Собираем пути к файлам выбранных кадров
    selected_files = [seq_dir / f"{sequence}_{frame:03d}.png" for frame in selected_frames]
    
    return selected_frames, selected_files


def create_frames_for_gif(frames, slow_factor=1):
    """
    Подготавливает кадры для плавной GIF анимации
    
    Args:
        frames: список кадров
        slow_factor: коэффициент замедления (повторяет каждый кадр slow_factor раз)
        
    Returns:
        list: список кадров с учетом замедления
    """
    if slow_factor <= 1:
        return frames
    
    repeated_frames = []
    for frame in frames:
        repeated_frames.extend([frame] * slow_factor)
    
    return repeated_frames


def save_gif_animation(frames_paths, output_path, fps=5, loop=0):
    """
    Создает GIF анимацию из набора изображений
    
    Args:
        frames_paths: список путей к изображениям
        output_path: путь для сохранения GIF
        fps: количество кадров в секунду
        loop: количество повторов (0 = бесконечно)
        
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    try:
        if not frames_paths:
            print("Ошибка: Нет кадров для создания GIF")
            return False
            
        # Сортируем кадры, чтобы убедиться, что они в правильном порядке
        frames_paths.sort()
        
        print(f"Создание GIF из {len(frames_paths)} кадров...")
        
        # Читаем все кадры и определяем их размеры
        pil_frames = []
        frame_sizes = []
        
        for i, frame_path in enumerate(frames_paths):
            try:
                # Загружаем как PIL изображение для более надежного управления размером
                pil_img = Image.open(frame_path)
                
                # Проверяем размер
                frame_sizes.append(pil_img.size)
                
                # Приводим к RGB если нужно
                if pil_img.mode == 'RGBA':
                    # Обрабатываем прозрачность, заменяя на белый
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])  # 3 is the alpha channel
                    pil_img = background
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                pil_frames.append(pil_img)
                
                # Для индикации прогресса
                if i % 20 == 0 and i > 0:
                    print(f"Загружено {i}/{len(frames_paths)} кадров")
                    
            except Exception as e:
                print(f"Ошибка загрузки кадра {i} ({frame_path}): {e}")
                import traceback
                traceback.print_exc()
        
        if not pil_frames:
            print("Ошибка: Не удалось загрузить ни одного кадра")
            return False
            
        # Определение согласованного размера для всех кадров
        print("Анализ размеров кадров...")
        
        # Используем первый кадр как эталон размера, чтобы избежать колебаний размеров
        # Это самый надежный способ обеспечить стабильность размеров между кадрами
        standard_width, standard_height = frame_sizes[0]
        
        print(f"Выбран стандартный размер кадра: {standard_width}x{standard_height}")
        
        # Убеждаемся, что размеры четные для лучшей совместимости
        standard_width = standard_width - (standard_width % 2)
        standard_height = standard_height - (standard_height % 2)
        
        standard_size = (standard_width, standard_height)
        
        # Приводим все кадры к одному размеру
        uniform_frames = []
        for i, pil_img in enumerate(pil_frames):
            # Всегда приводим к стандартному размеру, даже если размеры совпадают
            # Это гарантирует единообразие всех кадров
            if pil_img.size != standard_size:
                resized_img = pil_img.resize(standard_size, Image.LANCZOS)
                uniform_frames.append(resized_img)
                print(f"Изменен размер кадра {i} с {pil_img.size} до {standard_size}")
            else:
                uniform_frames.append(pil_img)
        
        # Сохраняем GIF с фиксированными настройками
        print(f"Сохраняем GIF с {len(uniform_frames)} кадрами (FPS={fps})")
        
        # Важные настройки, которые делают GIF стабильным:
        # - optimize=False предотвращает автоматическую оптимизацию, которая может изменить размеры
        # - disposal=2 указывает, что каждый кадр должен полностью заменять предыдущий
        uniform_frames[0].save(
            output_path,
            format='GIF',
            append_images=uniform_frames[1:],
            save_all=True,
            duration=int(1000/fps),  # продолжительность в мс
            loop=loop,
            optimize=False,
            disposal=2  # Каждый кадр полностью заменяет предыдущий
        )
        
        print(f"Анимация сохранена в {output_path}")
        return True
    except Exception as e:
        import traceback
        print(f"Ошибка при создании GIF: {str(e)}")
        traceback.print_exc()
        return False


def create_temp_directory(output_path, temp_dir_name="tmp_frames"):
    """
    Создает временную директорию для хранения кадров
    
    Args:
        output_path: путь к выходному файлу
        temp_dir_name: имя для временной директории
        
    Returns:
        Path: путь к временной директории
    """
    tmp_dir = Path(output_path).parent / temp_dir_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def cleanup_temp_directory(tmp_dir):
    """
    Удаляет временную директорию
    
    Args:
        tmp_dir: путь к временной директории
        
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    try:
        shutil.rmtree(tmp_dir)
        return True
    except Exception as e:
        print(f"Предупреждение: Не удалось удалить временную директорию: {e}")
        return False


# =====================================================================
# Функции для работы с данными и файлами
# =====================================================================

def load_ground_truth(gt_path):
    """
    Загружает ground truth данные из JSONL файла
    
    Args:
        gt_path: путь к файлу gt.jsonl
        
    Returns:
        dict: словарь {имя_изображения: {'bbox': [...], 'class': '...'}}
    """
    gt_data = {}
    try:
        print(f"Загрузка ground truth данных из {gt_path}...")
        with open(gt_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Поддержка разных форматов JSONL
                img_id = data.get('image_id', data.get('img', ''))
                bbox = data.get('bbox', [0, 0, 0, 0])
                class_id = data.get('category_id', data.get('class', 1))
                
                # Получаем компоненты имени
                parts = img_id.split('_')
                if len(parts) >= 3:
                    # Для последовательностей типа seq_0_XX
                    seq_name = parts[0]
                    seq_num = parts[1]
                    frame_num = int(parts[2])
                    
                    # Создаем сопоставление для всех возможных вариантов
                    # 1. seq_0_00.png (с двумя цифрами)
                    img_filename_2digits = f"{seq_name}_{seq_num}_{frame_num:02d}.png"
                    # 2. seq_0_000.png (с тремя цифрами)
                    img_filename_3digits = f"{seq_name}_{seq_num}_{frame_num:03d}.png"
                    # 3. Оригинальное имя (seq_0_00) плюс .png
                    img_filename_original = f"{img_id}.png"
                    # 4. Только оригинальное имя без .png (для некоторых случаев)
                    img_filename_no_ext = img_id
                    
                    # Сохраняем данные для всех вариантов имен файлов
                    for filename in [img_filename_2digits, img_filename_3digits, 
                                     img_filename_original, img_filename_no_ext]:
                        gt_data[filename] = {
                            'bbox': bbox,
                            'class': str(class_id)
                        }
                else:
                    # Если формат другой, используем стандартное преобразование
                    img_filename = img_id
                    if not img_id.endswith('.png') and not img_id.endswith('.jpg'):
                        img_filename = f"{img_id}.png"
                    
                    gt_data[img_filename] = {
                        'bbox': bbox,
                        'class': str(class_id)
                    }
                    gt_data[img_id] = {
                        'bbox': bbox,
                        'class': str(class_id)
                    }
        
        if not gt_data:
            print("Предупреждение: Ground truth данные загружены, но словарь пуст")
        else:
            print(f"Загружено {len(gt_data)/4} объектов ground truth") # Делим на 4, так как для каждого объекта у нас 4 варианта ключей
        
        return gt_data
    except Exception as e:
        print(f"Ошибка при загрузке ground truth данных: {e}")
        return {}


def load_csv_results(csv_path):
    """
    Загружает результаты из CSV файла
    
    Args:
        csv_path: путь к CSV файлу с результатами
        
    Returns:
        pd.DataFrame: данные из CSV или None в случае ошибки
    """
    try:
        if not os.path.exists(csv_path):
            print(f"Ошибка: Файл {csv_path} не найден")
            return None
        
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке CSV данных из {csv_path}: {e}")
        return None


def parse_bbox_string(bbox_str):
    """
    Преобразует строковое представление bbox в список
    
    Args:
        bbox_str: строка с bbox в формате '[x, y, w, h]' или 'x,y,w,h' или 
                 "\"x,y,w,h\"" (из CSV) или уже список/кортеж
        
    Returns:
        list: [x, y, w, h]
    """
    if isinstance(bbox_str, (list, tuple, np.ndarray)):
        return list(bbox_str)
    
    if isinstance(bbox_str, str):
        try:
            # Удаляем лишние кавычки, если они есть
            clean_str = bbox_str.strip('"\'')
            
            # Проверяем, содержит ли строка запятые без скобок (формат "x,y,w,h")
            if ',' in clean_str and not ('[' in clean_str or '(' in clean_str):
                return [float(x.strip()) for x in clean_str.split(',')]
            # Проверяем, может ли это быть список в строковом представлении (формат "[x, y, w, h]")
            elif '[' in clean_str or '(' in clean_str:
                return eval(clean_str)
            else:
                print(f"Предупреждение: Необычный формат bbox: {bbox_str}, возвращаем пустой bbox")
                return [0, 0, 0, 0]
        except Exception as e:
            print(f"Ошибка при разборе bbox {bbox_str}: {e}")
            return [0, 0, 0, 0]
    
    # Для случая, если передано число (некорректный формат)
    if isinstance(bbox_str, (int, float)):
        print(f"Предупреждение: Получено числовое значение вместо bbox: {bbox_str}")
        return [0, 0, 0, 0]
    
    # Для других типов данных
    print(f"Предупреждение: Неизвестный формат bbox: {bbox_str}, тип: {type(bbox_str)}")
    return [0, 0, 0, 0]


def find_sequence_ids(seq_dir):
    """
    Находит все идентификаторы последовательностей в директории
    
    Args:
        seq_dir: директория с последовательностями
        
    Returns:
        list: список идентификаторов последовательностей
    """
    seq_dir = Path(seq_dir)
    sequences = set()
    
    for img_path in seq_dir.glob("seq_*_*.png"):
        parts = img_path.stem.split('_')
        if len(parts) >= 2:
            seq_id = f"seq_{parts[1]}"
            sequences.add(seq_id)
    
    return sorted(list(sequences))


def generate_tips_results(baseline_csv, output_csv, window_size=5, confidence_boost=0.02):
    """
    ОПАСНО: ЭТА ФУНКЦИЯ БОЛЬШЕ НЕ ИСПОЛЬЗУЕТСЯ.
    Эта функция искусственно модифицировала результаты детекции, что недопустимо.
    Вместо этого используйте run_tips_yolo для честной реализации TIPS.
    
    Для корректной реализации TIPS необходимо:
    1. Применить небольшие сдвиги к входному изображению при инференсе
    2. Выполнить предсказания для каждой смещенной версии
    3. Выровнять координаты обратно в исходное пространство
    4. Агрегировать результаты через NMS или усреднение
    
    См. функцию run_tips_yolo, которая реализует этот подход корректно.
    """
    print("ПРЕДУПРЕЖДЕНИЕ: Функция generate_tips_results больше не используется!")
    print("Используйте run_tips_yolo для честной реализации TIPS")
    return None


def run_tips_yolo(model, image_path, shifts=[(0,0), (1,0), (0,1), (-1,0), (0,-1)], conf_threshold=0.25):
    """
    Применяет TIPS (Test-time Image-based Processing for Shift-invariance) к модели YOLO.
    TIPS выполняет предсказания на нескольких слегка смещенных версиях изображения
    и объединяет результаты для повышения стабильности.
    
    Args:
        model: предварительно загруженная модель YOLO
        image_path: путь к изображению для детекции
        shifts: список сдвигов (dx, dy) для применения к изображению 
                по умолчанию: [(0,0), (1,0), (0,1), (-1,0), (0,-1)]
        conf_threshold: порог уверенности для фильтрации детекций
        
    Returns:
        dict: агрегированные результаты детекции
    """
    import cv2
    import numpy as np
    from torchvision.ops import nms
    import torch
    
    # Загружаем изображение
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Не удалось загрузить изображение {image_path}")
        return None
    
    height, width = original_image.shape[:2]
    
    # Подготавливаем структуры для сбора всех результатов
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    # Применяем каждый сдвиг и выполняем детекцию
    for dx, dy in shifts:
        # Создаем матрицу преобразования для сдвига
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_image = cv2.warpAffine(original_image, M, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
        
        # Запускаем модель на сдвинутом изображении
        results = model(shifted_image, verbose=False)
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = boxes.conf[i].item()
                
                # Фильтруем по порогу уверенности
                if confidence < conf_threshold:
                    continue
                
                # Получаем координаты в формате [x1, y1, x2, y2]
                xyxy = boxes.xyxy[i].cpu().numpy()
                
                # Компенсируем изначальный сдвиг, чтобы вернуться в оригинальное пространство координат
                xyxy[0] -= dx
                xyxy[2] -= dx
                xyxy[1] -= dy
                xyxy[3] -= dy
                
                # Проверяем, что бокс не вышел за границы изображения
                xyxy[0] = max(0, xyxy[0])
                xyxy[1] = max(0, xyxy[1])
                xyxy[2] = min(width, xyxy[2])
                xyxy[3] = min(height, xyxy[3])
                
                # Собираем все детекции
                all_boxes.append(xyxy)
                all_scores.append(confidence)
                all_class_ids.append(cls_id)
    
    # Если детекций нет
    if not all_boxes:
        return None
    
    # Конвертируем в тензоры для NMS
    boxes = torch.tensor(all_boxes, dtype=torch.float32)
    scores = torch.tensor(all_scores, dtype=torch.float32)
    class_ids = torch.tensor(all_class_ids, dtype=torch.int64)
    
    # Агрегируем результаты, используя NMS
    # Сначала группируем по классам
    unique_classes = class_ids.unique()
    final_boxes = []
    final_scores = []
    final_class_ids = []
    
    for cls in unique_classes:
        # Фильтруем детекции для текущего класса
        indices = (class_ids == cls).nonzero(as_tuple=True)[0]
        cls_boxes = boxes[indices]
        cls_scores = scores[indices]
        
        # Применяем NMS
        keep_indices = nms(cls_boxes, cls_scores, iou_threshold=0.45)
        
        # Добавляем сохраненные детекции
        final_boxes.extend(cls_boxes[keep_indices].tolist())
        final_scores.extend(cls_scores[keep_indices].tolist())
        final_class_ids.extend([cls.item()] * len(keep_indices))
    
    # Конвертируем в формат [x, y, w, h]
    final_xywh_boxes = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        final_xywh_boxes.append([x1, y1, w, h])
    
    # Создаем объект-имитацию результатов модели YOLO для совместимости с остальным кодом
    class TIPSDetectionResult:
        def __init__(self, boxes, scores, class_ids, class_names):
            self.boxes = boxes
            self.scores = scores
            self.class_ids = class_ids
            self.class_names = class_names
    
    class TIPSBoxes:
        def __init__(self, xywh, xyxy, conf, cls, class_names):
            self.xywh = xywh
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls
            self.names = class_names
    
    # Создаем тензоры для боксов
    xywh = torch.tensor(final_xywh_boxes, dtype=torch.float32)
    xyxy = torch.tensor(final_boxes, dtype=torch.float32)
    conf = torch.tensor(final_scores, dtype=torch.float32)
    cls = torch.tensor(final_class_ids, dtype=torch.int64)
    
    # Формируем результат
    boxes = TIPSBoxes(
        xywh=xywh, 
        xyxy=xyxy, 
        conf=conf, 
        cls=cls, 
        class_names=model.names
    )
    
    result = TIPSDetectionResult(
        boxes=boxes,
        scores=final_scores,
        class_ids=final_class_ids,
        class_names=model.names
    )
    
    return [result]


def load_tips_yolo_model(weights_path, device='cpu'):
    """
    Загружает модель YOLO и оборачивает ее функцией TIPS для инференса.
    
    Args:
        weights_path: путь к весам модели YOLO
        device: устройство для выполнения (cpu, cuda, mps)
        
    Returns:
        callable: функция для запуска модели с TIPS
    """
    from ultralytics import YOLO
    
    try:
        # Загружаем базовую модель
        base_model = YOLO(weights_path)
        
        # Создаем обертку, которая использует TIPS при инференсе
        def tips_model(img, **kwargs):
            return run_tips_yolo(base_model, img, **kwargs)
        
        # Копируем некоторые нужные атрибуты из оригинальной модели
        tips_model.names = base_model.names
        
        return tips_model
    except Exception as e:
        print(f"Ошибка при загрузке TIPS-YOLO модели: {e}")
        return None 