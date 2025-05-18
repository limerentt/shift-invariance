#!/usr/bin/env python3
"""
Скрипт для обновления CSV-файлов с результатами классификаторов.
Версия для тестирования, которая не требует PyTorch и добавляет случайные
значения confidence вместо запуска реальной модели.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

# Для импорта из родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# Функции для обновления CSV-файлов
# =====================================================================

def update_csv_with_direct_confidence(csv_path, seq_dir):
    """
    Обновляет CSV-файл, добавляя метрику confidence, полученную из файлов 
    с реальными значениями или вычисленную из существующих метрик 
    
    Args:
        csv_path: путь к CSV-файлу
        seq_dir: директория с последовательностями изображений
        
    Returns:
        bool: True, если обновление успешно
    """
    try:
        # Загружаем CSV-файл
        df = pd.read_csv(csv_path)
        
        # Проверяем, есть ли уже столбец confidence
        if 'confidence' in df.columns:
            # Проверим, не была ли confidence получена из conf_drift
            if 'conf_drift' in df.columns:
                # Проверим несколько строк, чтобы определить, была ли confidence вычислена из conf_drift
                sample_rows = min(5, len(df))
                is_derived = all(abs(df['confidence'].iloc[i] - (1.0 - df['conf_drift'].iloc[i])) < 1e-6 
                                for i in range(sample_rows))
                
                if is_derived:
                    print(f"Столбец confidence был получен из conf_drift. Заменяем на реальные значения...")
                else:
                    print(f"Столбец confidence уже содержит прямые значения в {csv_path}, пропускаем...")
                    return True
            else:
                print(f"Столбец confidence уже существует в {csv_path}, пропускаем...")
                return True
        
        # Получаем название модели и последовательности из имени файла
        filename = os.path.basename(csv_path)
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"Неверный формат имени файла: {filename}")
            return False
        
        model_name = parts[0]
        sequence = "_".join(parts[1:]).replace('.csv', '')
        
        # Путь к файлу с реальными значениями confidence
        real_conf_path = Path(seq_dir).parent / "results" / "real_confidence" / f"{model_name}_{sequence}.txt"
        
        if real_conf_path.exists():
            # Загружаем реальные значения confidence
            try:
                real_conf = pd.read_csv(real_conf_path, header=None).values.flatten()
                if len(real_conf) != len(df):
                    print(f"Ошибка: несоответствие длин ({len(real_conf)} vs {len(df)}) в {real_conf_path}")
                    # Используем альтернативный подход - вычисляем из других метрик
                    if 'conf_drift' in df.columns:
                        print(f"Вычисляем confidence из conf_drift для {csv_path}")
                        df['confidence'] = 1.0 - df['conf_drift']
                    else:
                        print(f"Нет данных для вычисления confidence в {csv_path}")
                        return False
                else:
                    # Используем реальные значения
                    df['confidence'] = real_conf
                    print(f"Добавлены реальные значения confidence из {real_conf_path}")
            except Exception as e:
                print(f"Ошибка при чтении реальных значений confidence: {e}")
                # Используем альтернативный подход
                if 'conf_drift' in df.columns:
                    print(f"Вычисляем confidence из conf_drift для {csv_path}")
                    df['confidence'] = 1.0 - df['conf_drift']
                else:
                    print(f"Нет данных для вычисления confidence в {csv_path}")
                    return False
        else:
            # Если нет файла с реальными значениями, вычисляем из существующих метрик
            if 'conf_drift' in df.columns:
                print(f"Файл с реальными значениями не найден: {real_conf_path}")
                print(f"Вычисляем confidence из conf_drift для {csv_path}")
                df['confidence'] = 1.0 - df['conf_drift']
            else:
                print(f"Нет данных для вычисления confidence в {csv_path}")
                return False
        
        # Сохраняем обновленный CSV-файл
        df.to_csv(csv_path, index=False)
        
        print(f"CSV-файл успешно обновлен со значениями confidence: {csv_path}")
        return True
    
    except Exception as e:
        print(f"Ошибка при обновлении CSV-файла {csv_path}: {e}")
        return False

# =====================================================================
# Основные функции
# =====================================================================

def find_sequence_ids(seq_dir):
    """
    Находит все уникальные идентификаторы последовательностей в директории
    
    Args:
        seq_dir: директория с изображениями последовательностей
        
    Returns:
        list: список идентификаторов последовательностей (seq_0, seq_1, ...)
    """
    sequence_ids = set()
    
    # Ищем файлы по шаблону seq_X_XXX.png
    pattern = re.compile(r'(seq_\d+)_\d+\.png')
    
    for file_name in os.listdir(seq_dir):
        if file_name.endswith('.png'):
            match = pattern.match(file_name)
            if match:
                sequence_ids.add(match.group(1))
    
    return sorted(list(sequence_ids))

def process_model_type(model_type, results_dir, seq_dir, sequences=None):
    """
    Обрабатывает модели одного типа (ResNet или VGG)
    
    Args:
        model_type: тип модели ("resnet" или "vgg")
        results_dir: директория с CSV-файлами результатов
        seq_dir: директория с последовательностями изображений
        sequences: список ID последовательностей (если None, обрабатываются все)
        
    Returns:
        bool: True, если обработка успешна
    """
    # Определяем имена моделей в зависимости от типа
    if model_type == "resnet":
        model_names = ["ResNet50", "AA-ResNet50", "TIPS-ResNet50"]
    elif model_type == "vgg":
        model_names = ["VGG16", "AA-VGG16", "TIPS-VGG16"]
    else:
        print(f"Неподдерживаемый тип модели: {model_type}")
        return False
    
    # Обрабатываем каждую модель и последовательность
    for model_name in model_names:
        for sequence in sequences:
            csv_path = Path(results_dir) / f"{model_name}_{sequence}.csv"
            
            if not csv_path.exists():
                print(f"Файл не найден: {csv_path}")
                continue
            
            print(f"Обработка {model_name} для последовательности {sequence}...")
            update_csv_with_direct_confidence(csv_path, seq_dir)
    
    return True


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Обновление CSV-файлов с результатами классификаторов: добавление тестовых значений confidence"
    )
    
    # Обязательные аргументы
    parser.add_argument(
        "--seq-dir", type=str, required=True, 
        help="Директория с последовательностями изображений"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Директория с CSV-файлами результатов"
    )
    
    # Опциональные аргументы
    parser.add_argument(
        "--model-type", type=str, choices=["resnet", "vgg", "all"], default="all", 
        help="Тип моделей для обработки"
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="ID конкретной последовательности для обработки (если None, обрабатываются все)"
    )
    
    return parser.parse_args()


def main():
    """Основная функция"""
    args = parse_args()
    
    # Проверяем существование директорий
    if not os.path.exists(args.seq_dir):
        print(f"Директория с последовательностями не найдена: {args.seq_dir}")
        return False
    
    if not os.path.exists(args.results_dir):
        print(f"Директория с результатами не найдена: {args.results_dir}")
        return False
    
    # Получаем список последовательностей
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = find_sequence_ids(args.seq_dir)
    
    print(f"Обнаружены последовательности: {sequences}")
    
    # Обрабатываем модели в зависимости от типа
    if args.model_type == "all":
        process_model_type("resnet", args.results_dir, args.seq_dir, sequences)
        process_model_type("vgg", args.results_dir, args.seq_dir, sequences)
    else:
        process_model_type(args.model_type, args.results_dir, args.seq_dir, sequences)
    
    print("\nОбработка успешно завершена!")
    return True


if __name__ == "__main__":
    main() 