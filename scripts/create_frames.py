#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для создания тестовых кадров последовательностей для визуализации bounding box'ов.
Генерирует изображения со сдвигом объекта для демонстрации субпиксельных сдвигов.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math

# Создаем директории для последовательностей
os.makedirs("data/sequences", exist_ok=True)

# Определяем последовательности
sequences = ["seq_0", "seq_1", "seq_2"]

# Параметры изображений
width, height = 400, 300  # размер изображения
object_size = 50  # размер объекта
num_frames = 32  # количество кадров в последовательности

# Цвета объектов для разных последовательностей (в формате BGR для OpenCV)
colors = {
    "seq_0": (0, 0, 255),    # красный
    "seq_1": (0, 255, 0),    # зеленый
    "seq_2": (255, 0, 0)     # синий
}

def create_sequence_frames(seq_name, bg_color=(240, 240, 240)):
    """
    Создает последовательность кадров с перемещающимся объектом.
    
    Args:
        seq_name: имя последовательности (seq_0, seq_1, seq_2)
        bg_color: цвет фона (в формате BGR для OpenCV)
    """
    print(f"Создание кадров для последовательности {seq_name}...")
    
    # Выбираем цвет объекта
    obj_color = colors[seq_name]
    
    # Начальная позиция объекта (центр)
    start_x = width // 2 - object_size // 2
    start_y = height // 2 - object_size // 2
    
    # Создаем кадры с перемещением объекта
    for i in range(num_frames):
        # Вычисляем смещение для текущего кадра (1 пиксель горизонтально)
        offset_x = i
        
        # Создаем пустое изображение с фоном
        img = np.ones((height, width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        
        # Вычисляем позицию объекта с учетом смещения
        x = start_x + offset_x
        y = start_y
        
        # Рисуем объект (прямоугольник)
        cv2.rectangle(img, (x, y), (x + object_size, y + object_size), obj_color, -1)
        
        # Добавляем номер кадра как текст
        cv2.putText(img, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Сохраняем изображение
        filename = f"data/sequences/{seq_name}_{i:02d}.png"
        cv2.imwrite(filename, img)
    
    print(f"Создано {num_frames} кадров для последовательности {seq_name}")

def create_all_sequences():
    """Создает кадры для всех последовательностей."""
    for seq_name in sequences:
        create_sequence_frames(seq_name)

if __name__ == "__main__":
    print("Генерация кадров для визуализации bounding box'ов...")
    create_all_sequences()
    print("Готово! Все кадры созданы.") 