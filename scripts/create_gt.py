#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для создания файла ground truth в формате JSONL.
Генерирует аннотации bbox'ов для каждого кадра последовательностей.
"""

import os
import json
import math
from pathlib import Path

# Создаем директорию, если она не существует
os.makedirs("data/sequences", exist_ok=True)

# Определяем последовательности
sequences = ["seq_0", "seq_1", "seq_2"]

# Параметры изображений (должны совпадать с create_frames.py)
width, height = 400, 300  # размер изображения
object_size = 50  # размер объекта
num_frames = 32  # количество кадров в последовательности

# Начальная позиция объекта (должна совпадать с create_frames.py)
start_x = width // 2 - object_size // 2
start_y = height // 2 - object_size // 2

def create_ground_truth():
    """
    Создает файл ground truth в формате JSONL.
    Каждая строка содержит аннотацию для одного кадра.
    """
    print("Создание файла ground truth...")
    
    gt_entries = []
    
    for seq_name in sequences:
        for i in range(num_frames):
            # Вычисляем смещение для текущего кадра (1 пиксель горизонтально)
            offset_x = i
            
            # Вычисляем позицию объекта с учетом смещения
            x = start_x + offset_x
            y = start_y
            
            # Создаем запись ground truth
            gt_entry = {
                "image_id": f"{seq_name}_{i:02d}",
                "bbox": [x, y, object_size, object_size],  # [x, y, width, height]
                "category_id": 1,  # Присваиваем категорию 1 (объект)
                "area": object_size * object_size,
                "iscrowd": 0
            }
            
            gt_entries.append(gt_entry)
    
    # Сохраняем в JSONL формате (каждая строка - один JSON объект)
    output_path = "data/sequences/gt.jsonl"
    with open(output_path, 'w') as f:
        for entry in gt_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Создан файл ground truth: {output_path}")
    print(f"Всего аннотаций: {len(gt_entries)}")

if __name__ == "__main__":
    create_ground_truth() 