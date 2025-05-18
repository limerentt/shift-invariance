#!/usr/bin/env python3
"""
Модуль с функциями для работы с детекторами объектов (YOLO и др.)
и обработки результатов детекции.
"""

import csv
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Импортируем общие утилиты
from scripts.utils import calculate_iou, calculate_center_shift, load_ground_truth


def load_yolo_model(weights_path, device='cpu'):
    """
    Загружает модель YOLO из весов
    
    Args:
        weights_path: путь к весам модели (.pt)
        device: устройство для загрузки модели (cpu, cuda, mps)
        
    Returns:
        model: загруженная модель YOLO
    """
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели YOLO: {e}")
        return None


def load_aa_yolo_model(weights_path, device='cpu'):
    """
    Загружает anti-aliased версию модели YOLO, модифицируя архитектуру
    с использованием библиотеки antialiased-cnns от Adobe Research.
    
    Anti-aliasing реализуется через добавление low-pass фильтров (BlurPool)
    перед операциями subsampling для предотвращения алиасинга в соответствии 
    с статьей "Making Convolutional Networks Shift-Invariant Again".
    
    Args:
        weights_path: путь к весам модели (.pt)
        device: устройство для загрузки модели (cpu, cuda, mps)
        
    Returns:
        model: загруженная anti-aliased модель YOLO
    """
    try:
        import torch
        import torch.nn as nn
        from ultralytics import YOLO
        import antialiased_cnns as aa
        
        # Загружаем базовую модель
        base_model = YOLO(weights_path)
        model = base_model.model
        
        # Создаем собственную антиалиасную convolution, комбинируя обычную Conv2d и BlurPool
        class AntiAliasedConv2d(nn.Module):
            def __init__(self, conv, filt_size=3, pad_type='reflect'):
                super(AntiAliasedConv2d, self).__init__()
                
                # Извлекаем параметры из оригинальной convolution
                stride = conv.stride
                
                # Создаем новую convolution с stride=1
                self.conv = nn.Conv2d(
                    in_channels=conv.in_channels,
                    out_channels=conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=1,  # Всегда используем stride=1
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=conv.bias is not None
                )
                
                # Копируем веса из оригинальной convolution
                with torch.no_grad():
                    self.conv.weight.copy_(conv.weight)
                    if conv.bias is not None:
                        self.conv.bias.copy_(conv.bias)
                
                # Создаем blur pool с нужным stride
                if isinstance(stride, int):
                    blur_stride = stride
                else:
                    blur_stride = stride[0]  # Берем первый элемент, если кортеж
                
                # Создаем blur pool с правильными параметрами
                self.blur = aa.BlurPool(
                    channels=int(conv.out_channels),  # Убеждаемся, что это int
                    stride=blur_stride,
                    filt_size=filt_size,
                    pad_type=pad_type
                )
            
            def forward(self, x):
                x = self.conv(x)
                x = self.blur(x)
                return x
        
        # Рекурсивная функция для замены операций downsampling на их anti-aliased версии
        def replace_modules_with_aa(module, parent_name=''):
            for name, child in list(module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                # Если это составной модуль, рекурсивно обрабатываем его
                if len(list(child.children())) > 0:
                    replace_modules_with_aa(child, full_name)
                
                # Заменяем MaxPool на BlurPool
                if isinstance(child, nn.MaxPool2d):
                    # Определяем число каналов из предыдущего слоя
                    # (это сложная задача в общем случае, поэтому аппроксимируем)
                    try:
                        prev_layer = None
                        for prev_name, prev_mod in module.named_children():
                            if prev_name == name:
                                break
                            prev_layer = prev_mod
                        
                        channels = 64  # значение по умолчанию
                        if hasattr(prev_layer, 'out_channels'):
                            channels = int(prev_layer.out_channels)
                        elif hasattr(prev_layer, 'out_features'):
                            channels = int(prev_layer.out_features)
                    except:
                        # Если не можем определить, берем значение по умолчанию
                        channels = 64  # стандартное количество каналов
                    
                    # Определяем stride
                    if hasattr(child, 'stride'):
                        stride = child.stride if isinstance(child.stride, int) else child.stride[0]
                    else:
                        stride = 2  # стандартный stride для MaxPool2d
                    
                    # Создаем anti-aliased max pool
                    try:
                        aa_maxpool = nn.Sequential(
                            nn.MaxPool2d(kernel_size=child.kernel_size if hasattr(child, 'kernel_size') else 2, 
                                         stride=1, 
                                         padding=child.padding if hasattr(child, 'padding') else 0),
                            aa.BlurPool(channels=channels, stride=stride, filt_size=3, pad_type='reflect')
                        )
                        
                        # Устанавливаем новый модуль вместо старого
                        setattr(module, name, aa_maxpool)
                        print(f"✓ Заменен MaxPool на BlurPool в {full_name}")
                    except Exception as e:
                        print(f"! Не удалось заменить MaxPool в {full_name}: {e}")
                
                # Заменяем Convolution с stride > 1 на anti-aliased версию
                elif isinstance(child, nn.Conv2d) and (
                        hasattr(child, 'stride') and 
                        (isinstance(child.stride, int) and child.stride > 1 or 
                         isinstance(child.stride, tuple) and max(child.stride) > 1)):
                    
                    try:
                        # Создаем anti-aliased convolution
                        aa_conv = AntiAliasedConv2d(conv=child, filt_size=3, pad_type='reflect')
                        
                        # Устанавливаем новый модуль вместо старого
                        setattr(module, name, aa_conv)
                        print(f"✓ Заменен Conv2d со stride={child.stride} на AntiAliasedConv2d в {full_name}")
                    except Exception as e:
                        print(f"! Не удалось заменить Conv2d в {full_name}: {e}")
        
        # Применяем модификацию ко всей модели
        print("Модифицируем архитектуру YOLO, добавляя anti-aliasing...")
        replace_modules_with_aa(model)
        
        # Переносим модель на целевое устройство
        model.to(device)
        
        print("Загружена Anti-Aliased YOLO модель с научно-обоснованной архитектурой")
        return base_model
    
    except ImportError as e:
        print(f"Ошибка: не удалось импортировать библиотеку antialiased_cnns.")
        print(f"Пожалуйста, установите ее командой: pip install antialiased-cnns")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке AA-YOLO модели: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_tips_yolo_model(weights_path, device='cpu'):
    """
    Загружает TIPS-YOLO модель, которая применяет множественные сдвиги
    во время инференса для улучшения инвариантности к сдвигам.
    
    Args:
        weights_path: путь к весам базовой модели YOLO
        device: устройство для загрузки модели
        
    Returns:
        model: функция-обертка для TIPS-YOLO инференса
    """
    from scripts.utils import load_tips_yolo_model as load_impl
    
    try:
        tips_model = load_impl(weights_path, device)
        print("Загружена TIPS-YOLO модель с честной реализацией сдвигов")
        return tips_model
    except Exception as e:
        print(f"Ошибка при загрузке TIPS-YOLO модели: {e}")
        return None


def run_detector_on_sequences(
    model, seq_dir, out_dir, class_filter=None, 
    conf_threshold=0.25, file_prefix=""
):
    """
    Запускает модель детектора (YOLO) на последовательностях изображений
    и сохраняет результаты детекции
    
    Args:
        model: модель детектора (YOLO)
        seq_dir: директория с последовательностями изображений и gt.jsonl
        out_dir: директория для сохранения результатов
        class_filter: список классов для фильтрации (None = все)
        conf_threshold: порог уверенности для детекций
        file_prefix: префикс для имени файла результатов
        
    Returns:
        dict: результаты детекции по последовательностям
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seq_dir = Path(seq_dir)
    gt_path = seq_dir / "gt.jsonl"
    
    # Загружаем ground truth данные
    gt_data = load_ground_truth(gt_path)
    if not gt_data:
        print(f"Ошибка: Не удалось загрузить ground truth данные из {gt_path}")
        return {}
    
    # Группируем изображения по последовательностям
    sequences = {}
    for img_path in sorted(seq_dir.glob("*.png")):
        img_name = img_path.name
        # Извлекаем идентификатор последовательности
        # seq_0_001.png -> seq_0
        parts = img_name.split("_")
        if len(parts) >= 2:
            seq_id = "_".join(parts[:2])
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(img_path)
    
    if not sequences:
        print(f"Ошибка: Не найдены последовательности изображений в {seq_dir}")
        return {}
    
    # Результаты по последовательностям
    all_results = {}
    
    # Обрабатываем каждую последовательность
    for seq_id, img_paths in tqdm(sequences.items(), desc="Обработка последовательностей"):
        results = []
        
        for img_path in sorted(img_paths):
            img_name = img_path.name
            
            # Проверяем наличие ground truth данных
            if img_name not in gt_data:
                print(f"Предупреждение: Отсутствуют ground truth данные для {img_name}")
                continue
            
            # Запуск детектора
            detections = model(img_path, verbose=False)
            
            # Извлекаем Ground Truth
            gt_bbox = gt_data[img_name]['bbox']
            gt_class = gt_data[img_name]['class']
            
            # Выбираем лучшую детекцию (по IoU или принадлежности к ожидаемому классу)
            best_detection = None
            best_iou = -1
            
            for detection in detections:
                pred_boxes = detection.boxes
                
                for i in range(len(pred_boxes)):
                    cls_id = int(pred_boxes.cls[i].item())
                    cls_name = model.names[cls_id]
                    conf = pred_boxes.conf[i].item()
                    
                    # Пропускаем, если не проходит по классу или уверенности
                    if class_filter and cls_name not in class_filter:
                        continue
                    if conf < conf_threshold:
                        continue
                    
                    # Преобразуем bbox в формат [x, y, w, h] из [x1, y1, x2, y2]
                    xywh = pred_boxes.xywh[i].cpu().numpy()
                    x, y, w, h = xywh
                    pred_bbox = [float(x - w/2), float(y - h/2), float(w), float(h)]
                    
                    # Вычисляем IoU
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = {
                            'bbox': pred_bbox,
                            'class': cls_name,
                            'confidence': conf,
                            'iou': iou
                        }
            
            # Если детекция не найдена, создаем пустую
            if best_detection is None:
                best_detection = {
                    'bbox': [0, 0, 0, 0],
                    'class': 'none',
                    'confidence': 0.0,
                    'iou': 0.0
                }
            
            # Вычисляем дополнительные метрики
            center_shift = calculate_center_shift(best_detection['bbox'], gt_bbox)
            
            # Сохраняем результаты
            result = {
                'frame': img_name,
                'gt_bbox': gt_bbox,
                'gt_class': gt_class,
                'pred_bbox': best_detection['bbox'],
                'pred_class': best_detection['class'],
                'confidence': best_detection['confidence'],
                'iou': best_detection['iou'],
                'center_shift': center_shift
            }
            
            results.append(result)
        
        # Сохраняем результаты в CSV
        prefix_part = f"{file_prefix}_" if file_prefix else ""
        csv_path = out_dir / f"{prefix_part}{seq_id}.csv"
        
        save_detection_results(results, csv_path)
        
        # Сохраняем результаты в общий словарь
        all_results[seq_id] = results
        
        print(f"Сохранены результаты для {seq_id} в {csv_path}")
    
    return all_results


def save_detection_results(results, csv_path):
    """
    Сохраняет результаты детекции в CSV файл
    
    Args:
        results: список результатов детекции
        csv_path: путь для сохранения CSV
        
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame', 'gt_bbox', 'gt_class', 'pred_bbox', 'pred_class', 
                'confidence', 'iou', 'center_shift'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        return False


def compute_detection_metrics(results):
    """
    Вычисляет агрегированные метрики для результатов детекции
    
    Args:
        results: список результатов детекции
        
    Returns:
        dict: словарь с метриками
    """
    if not results:
        return {
            'avg_iou': 0.0,
            'std_iou': 0.0,
            'miss_rate': 1.0,
            'avg_center_shift': 0.0,
            'std_center_shift': 0.0,
            'avg_confidence': 0.0,
            'std_confidence': 0.0
        }
    
    # Извлекаем метрики из результатов
    ious = [r['iou'] for r in results]
    center_shifts = [r['center_shift'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Вычисляем метрики
    avg_iou = np.mean(ious)
    std_iou = np.std(ious)
    miss_rate = sum(1 for iou in ious if iou < 0.1) / len(ious)
    avg_center_shift = np.mean(center_shifts)
    std_center_shift = np.std(center_shifts)
    avg_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    
    return {
        'avg_iou': avg_iou,
        'std_iou': std_iou,
        'miss_rate': miss_rate,
        'avg_center_shift': avg_center_shift,
        'std_center_shift': std_center_shift,
        'avg_confidence': avg_confidence,
        'std_confidence': std_confidence
    }


def compute_metrics_for_all_sequences(results_dict):
    """
    Вычисляет метрики для всех последовательностей
    
    Args:
        results_dict: словарь {sequence_id: [результаты]} 
        
    Returns:
        dict: словарь {sequence_id: {метрики}}
    """
    metrics_dict = {}
    
    for seq_id, results in results_dict.items():
        metrics_dict[seq_id] = compute_detection_metrics(results)
    
    return metrics_dict


def process_single_sequence(model, seq_dir, seq_id, output_csv, gt_data=None, conf_threshold=0.25):
    """
    Обрабатывает одну последовательность изображений и сохраняет результаты
    
    Args:
        model: модель YOLO
        seq_dir: директория с последовательностями
        seq_id: идентификатор последовательности (например, seq_0)
        output_csv: путь к CSV файлу для сохранения результатов
        gt_data: данные ground truth (если None, будут загружены из seq_dir/gt.jsonl)
        conf_threshold: порог уверенности для детекций
        
    Returns:
        list: список результатов детекции для последовательности
    """
    seq_dir = Path(seq_dir)
    
    # Если ground truth не предоставлены, загружаем их
    if gt_data is None:
        gt_path = seq_dir / "gt.jsonl"
        gt_data = load_ground_truth(gt_path)
        if not gt_data:
            print(f"Ошибка: Не удалось загрузить ground truth данные из {gt_path}")
            return []
    
    # Находим изображения для данной последовательности
    img_paths = []
    for pattern in [
        f"{seq_id}_*.png",           # Например, seq_0_001.png
        f"{seq_id}/*.png",           # Например, seq_0/001.png
        f"{seq_id}*/*.png",          # Например, seq_00_00/001.png
    ]:
        img_paths.extend(seq_dir.glob(pattern))
    
    # Сортируем по имени для правильного порядка кадров
    img_paths = sorted(img_paths)
    
    if not img_paths:
        print(f"Ошибка: Не найдены изображения для последовательности {seq_id} в {seq_dir}")
        return []
    
    results = []
    
    # Обрабатываем каждое изображение в последовательности
    for img_path in img_paths:
        img_name = img_path.name
        
        # Проверяем наличие ground truth данных
        if img_name not in gt_data:
            print(f"Предупреждение: Отсутствуют ground truth данные для {img_name}")
            
            # Попробуем использовать только filename без path
            img_name_only = Path(img_name).name
            if img_name_only not in gt_data:
                # Если не нашли, используем пустые значения
                gt_bbox = [0, 0, 0, 0]
                gt_class = "none"
            else:
                # Используем ground truth для имени файла без пути
                gt_bbox = gt_data[img_name_only]['bbox']
                gt_class = gt_data[img_name_only]['class']
        else:
            # Используем найденные ground truth
            gt_bbox = gt_data[img_name]['bbox']
            gt_class = gt_data[img_name]['class']
        
        # Запуск детектора
        detections = model(img_path, verbose=False)
        
        # Выбираем лучшую детекцию (по IoU или уверенности)
        best_detection = None
        best_iou = -1
        
        for detection in detections:
            pred_boxes = detection.boxes
            
            for i in range(len(pred_boxes)):
                cls_id = int(pred_boxes.cls[i].item())
                cls_name = model.names[cls_id]
                conf = pred_boxes.conf[i].item()
                
                # Пропускаем, если не проходит по уверенности
                if conf < conf_threshold:
                    continue
                
                # Преобразуем bbox в формат [x, y, w, h] из [x1, y1, x2, y2]
                xywh = pred_boxes.xywh[i].cpu().numpy()
                x, y, w, h = xywh
                pred_bbox = [float(x - w/2), float(y - h/2), float(w), float(h)]
                
                # Вычисляем IoU
                iou = calculate_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_detection = {
                        'bbox': pred_bbox,
                        'class': cls_name,
                        'confidence': conf,
                        'iou': iou
                    }
        
        # Если детекция не найдена, создаем пустую
        if best_detection is None:
            best_detection = {
                'bbox': [0, 0, 0, 0],
                'class': 'none',
                'confidence': 0.0,
                'iou': 0.0
            }
        
        # Вычисляем дополнительные метрики
        center_shift = calculate_center_shift(best_detection['bbox'], gt_bbox)
        
        # Получаем номер кадра из имени файла
        # Например, seq_0_001.png -> 1
        try:
            parts = img_name.split('_')
            frame_num = int(parts[-1].replace('.png', ''))
        except:
            frame_num = len(results)  # Если не удалось извлечь номер, используем индекс
        
        # Сохраняем результаты
        result = {
            'frame': frame_num,
            'gt_bbox': gt_bbox,
            'gt_class': gt_class,
            'pred_bbox': best_detection['bbox'],
            'pred_class': best_detection['class'],
            'confidence': best_detection['confidence'],
            'iou': best_detection['iou'],
            'center_shift': center_shift
        }
        
        results.append(result)
    
    # Сохраняем результаты в CSV
    save_detection_results(results, output_csv)
    
    print(f"Сохранены результаты для {seq_id} в {output_csv}")
    
    return results


def compute_metrics(results):
    """
    Вычисляет метрики для результатов детекции
    
    Args:
        results: список результатов детекции
        
    Returns:
        dict: словарь с метриками
    """
    # Пропуски и ложные срабатывания
    n_frames = len(results)
    n_detections = sum(1 for r in results if r['confidence'] > 0.0)
    n_missed = n_frames - n_detections
    miss_rate = n_missed / n_frames if n_frames > 0 else 1.0
    
    # IoU
    ious = [r['iou'] for r in results if r['iou'] > 0.0]
    avg_iou = np.mean(ious) if ious else 0.0
    std_iou = np.std(ious) if len(ious) > 1 else 0.0
    
    # Смещение центра
    shifts = [r['center_shift'] for r in results if r['center_shift'] >= 0.0]
    avg_center_shift = np.mean(shifts) if shifts else 0.0
    std_center_shift = np.std(shifts) if len(shifts) > 1 else 0.0
    
    # Уверенность
    confidences = [r['confidence'] for r in results if r['confidence'] > 0.0]
    avg_confidence = np.mean(confidences) if confidences else 0.0
    std_confidence = np.std(confidences) if len(confidences) > 1 else 0.0
    
    return {
        'n_frames': n_frames,
        'n_detections': n_detections,
        'n_missed': n_missed,
        'miss_rate': miss_rate,
        'avg_iou': avg_iou,
        'std_iou': std_iou,
        'avg_center_shift': avg_center_shift,
        'std_center_shift': std_center_shift,
        'avg_confidence': avg_confidence,
        'std_confidence': std_confidence
    } 