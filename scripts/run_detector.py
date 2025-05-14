#!/usr/bin/env python3
"""
Интерфейс командной строки для запуска детекторов YOLO на последовательностях изображений
и сохранения результатов детекции.
"""

import os
import argparse
from pathlib import Path

from scripts.detection import load_yolo_model, run_detector_on_sequences
from scripts.utils import find_sequence_ids


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Запуск детекторов YOLO на последовательностях изображений"
    )
    
    # Обязательные аргументы
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Путь к весам модели YOLO (.pt)"
    )
    parser.add_argument(
        "--seq-dir", type=str, required=True,
        help="Директория с последовательностями изображений и gt.jsonl"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Директория для сохранения результатов"
    )
    
    # Опциональные аргументы
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Порог уверенности для детекций"
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Список классов через запятую (None = все)"
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Префикс для имени файла результатов"
    )
    parser.add_argument(
        "--compute-metrics", action="store_true",
        help="Вычислить и вывести метрики после детекции"
    )
    
    return parser.parse_args()


def main():
    """Основная функция"""
    args = parse_args()
    
    # Преобразуем пути в объекты Path
    weights_path = Path(args.weights)
    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    
    # Создаем выходную директорию, если её нет
    os.makedirs(out_dir, exist_ok=True)
    
    # Проверка наличия последовательностей
    sequences = find_sequence_ids(seq_dir)
    if not sequences:
        print(f"Ошибка: Не найдены последовательности в {seq_dir}")
        return
    
    print(f"Найдены последовательности: {sequences}")
    
    # Разбираем список классов
    class_filter = args.classes.split(',') if args.classes else None
    
    # Загружаем модель
    print(f"Загрузка модели из {weights_path}...")
    model = load_yolo_model(weights_path)
    if model is None:
        print("Ошибка: Не удалось загрузить модель")
        return
    
    # Запускаем детектор на последовательностях
    print(f"Запуск YOLO на последовательностях из {seq_dir}...")
    results = run_detector_on_sequences(
        model, 
        seq_dir, 
        out_dir, 
        class_filter, 
        args.conf, 
        args.prefix
    )
    
    # Выводим метрики
    if args.compute_metrics and results:
        print("\nМетрики детекции по последовательностям:")
        from scripts.detection import compute_metrics_for_all_sequences
        
        metrics = compute_metrics_for_all_sequences(results)
        for seq_id, seq_metrics in metrics.items():
            print(f"\n{seq_id}:")
            print(f"  IoU: {seq_metrics['avg_iou']:.4f} ± {seq_metrics['std_iou']:.4f}")
            print(f"  Пропуски: {seq_metrics['miss_rate']:.4f}")
            print(f"  Смещение центра: {seq_metrics['avg_center_shift']:.2f} ± {seq_metrics['std_center_shift']:.2f} px")
            print(f"  Уверенность: {seq_metrics['avg_confidence']*100:.2f}% ± {seq_metrics['std_confidence']*100:.2f}%")


if __name__ == "__main__":
    main() 