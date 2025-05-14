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


def load_yolo_model(weights_path):
    """
    Загружает модель YOLO из весов
    
    Args:
        weights_path: путь к весам модели (.pt)
        
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