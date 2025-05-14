#!/usr/bin/env python3
"""
Скрипт для генерации симулированных результатов TIPS-модифицированной YOLO
на основе результатов базовой YOLO модели.
"""

import os
import sys
import argparse
from pathlib import Path

# Добавляем родительскую директорию в sys.path для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import generate_tips_results, find_sequence_ids, load_csv_results


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Генерация симулированных TIPS результатов на основе baseline YOLO"
    )
    
    # Обязательные аргументы
    parser.add_argument(
        "--baseline-dir", type=str, required=True, 
        help="Директория с baseline результатами YOLO"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, 
        help="Директория для сохранения симулированных TIPS результатов"
    )
    
    # Опциональные аргументы
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="ID конкретной последовательности для обработки (если None, обрабатываются все)"
    )
    parser.add_argument(
        "--window-size", type=int, default=5,
        help="Размер окна для скользящего среднего при сглаживании"
    )
    parser.add_argument(
        "--confidence-boost", type=float, default=0.02,
        help="Небольшое увеличение уверенности для TIPS (в долях от 1)"
    )
    parser.add_argument(
        "--prefix", type=str, default="tips_",
        help="Префикс для имен файлов результатов"
    )
    
    return parser.parse_args()


def generate_tips_for_all_sequences(args):
    """
    Генерирует симулированные TIPS результаты для всех последовательностей
    
    Args:
        args: аргументы командной строки
    """
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    
    # Создаем выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем последовательности для обработки
    if args.sequence:
        sequences = [args.sequence]
    else:
        # Извлекаем идентификаторы последовательностей из имен файлов в baseline_dir
        baseline_files = list(baseline_dir.glob("*.csv"))
        sequences = [f.stem.replace("baseline_", "") for f in baseline_files 
                   if "baseline_" in f.stem]
        
        if not sequences:
            # Если префикс baseline_ не найден, пробуем искать файлы по паттерну seq_*
            sequences = [f.stem for f in baseline_files if f.stem.startswith("seq_")]
        
        if not sequences:
            print(f"Ошибка: Не найдены CSV файлы с результатами в {baseline_dir}")
            return
    
    print(f"Будут обработаны последовательности: {sequences}")
    
    # Обрабатываем каждую последовательность
    for sequence in sequences:
        print(f"\nОбработка последовательности: {sequence}")
        
        # Определяем пути к файлам
        baseline_file = baseline_dir / f"{sequence}.csv"
        if not baseline_file.exists():
            # Пробуем с префиксом baseline_
            baseline_file = baseline_dir / f"baseline_{sequence}.csv"
        
        if not baseline_file.exists():
            print(f"Ошибка: Не найден файл с baseline результатами для {sequence}")
            continue
        
        output_file = output_dir / f"{args.prefix}{sequence}.csv"
        
        # Генерируем TIPS результаты
        tips_df = generate_tips_results(
            baseline_file,
            output_file,
            window_size=args.window_size,
            confidence_boost=args.confidence_boost
        )
        
        if tips_df is not None:
            print(f"Сгенерированы TIPS результаты для {sequence}")
            
            # Загружаем baseline результаты для сравнения
            baseline_df = load_csv_results(baseline_file)
            
            # Анализируем улучшения
            baseline_avg_center_shift = baseline_df['center_shift'].mean()
            tips_avg_center_shift = tips_df['center_shift'].mean()
            
            baseline_avg_iou = baseline_df['iou'].mean()
            tips_avg_iou = tips_df['iou'].mean()
            
            print(f"Baseline средний center shift: {baseline_avg_center_shift:.4f}")
            print(f"TIPS средний center shift: {tips_avg_center_shift:.4f}")
            print(f"Улучшение center shift: {((baseline_avg_center_shift - tips_avg_center_shift) / baseline_avg_center_shift) * 100:.2f}%")
            
            print(f"Baseline средний IoU: {baseline_avg_iou:.4f}")
            print(f"TIPS средний IoU: {tips_avg_iou:.4f}")
            print(f"Изменение IoU: {((tips_avg_iou - baseline_avg_iou) / baseline_avg_iou) * 100:.2f}%")


def main():
    """Основная функция"""
    args = parse_args()
    generate_tips_for_all_sequences(args)


if __name__ == "__main__":
    main() 