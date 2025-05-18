#!/usr/bin/env python3
"""
Интерфейс командной строки для запуска детекторов YOLO на последовательностях изображений
и сохранения результатов детекции.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

from scripts.detection import (
    load_yolo_model, load_aa_yolo_model, load_tips_yolo_model,
    run_detector_on_sequences, load_ground_truth, process_single_sequence
)
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
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Устройство для загрузки модели (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--model-type", type=str, default="base",
        choices=["base", "aa", "tips"],
        help="Тип модели: base = стандартная YOLO, aa = anti-aliased YOLO, tips = TIPS-YOLO"
    )
    
    return parser.parse_args()


def process_sequence(model, seq_dir, seq_id, out_dir, prefix, compute_metrics=False, gt_data=None, conf_threshold=0.25):
    """
    Обрабатывает одну последовательность изображений
    
    Args:
        model: модель YOLO
        seq_dir: директория с последовательностями
        seq_id: идентификатор последовательности
        out_dir: директория для сохранения результатов
        prefix: префикс для имен выходных файлов
        compute_metrics: вычислять ли метрики
        gt_data: данные ground truth
        conf_threshold: порог уверенности для детекций
    """
    # Создаем выходную директорию
    os.makedirs(out_dir, exist_ok=True)
    
    # Имя выходного файла с результатами
    output_csv = os.path.join(out_dir, f"{prefix}_{seq_id}.csv")
    
    # Запускаем детектор на последовательности
    results = process_single_sequence(
        model, 
        os.path.join(seq_dir),
        seq_id,
        output_csv,
        gt_data=gt_data,
        conf_threshold=conf_threshold
    )
    
    # Выводим метрики
    if compute_metrics and results:
        from scripts.detection import compute_metrics
        
        metrics = compute_metrics(results)
        print(f"\n{seq_id}:")
        print(f"  IoU: {metrics['avg_iou']:.4f} ± {metrics['std_iou']:.4f}")
        print(f"  Пропуски: {metrics['miss_rate']:.4f}")
        print(f"  Смещение центра: {metrics['avg_center_shift']:.2f} ± {metrics['std_center_shift']:.2f} px")
        print(f"  Уверенность: {metrics['avg_confidence']*100:.2f}% ± {metrics['std_confidence']*100:.2f}%")


def main():
    """Основная функция для запуска детектора"""
    args = parse_args()
    
    # Находим последовательности
    sequences = find_sequence_ids(args.seq_dir)
    
    # Фильтруем последовательности, оставляем только те, для которых есть ground truth
    valid_sequences = []
    for seq in sequences:
        # Оставляем только последовательности seq_0, seq_1, seq_2
        if seq in ["seq_0", "seq_1", "seq_2"]:
            valid_sequences.append(seq)
    
    if not valid_sequences:
        print(f"Ошибка: Не найдены поддерживаемые последовательности в {args.seq_dir}")
        return
        
    print(f"Найдены последовательности: {valid_sequences}")
    
    # Грузим модель в зависимости от типа
    print(f"Загрузка модели типа '{args.model_type}' из {args.weights}...")
    model = None
    
    if args.model_type == "base":
        model = load_yolo_model(args.weights, device=args.device)
    elif args.model_type == "aa":
        model = load_aa_yolo_model(args.weights, device=args.device)
    elif args.model_type == "tips":
        model = load_tips_yolo_model(args.weights, device=args.device)
    else:
        print(f"Неизвестный тип модели: {args.model_type}")
        return
    
    if model is None:
        print(f"Не удалось загрузить модель из {args.weights}")
        return
    
    # Запускаем на всех последовательностях
    print(f"Запуск {args.model_type.upper()}-YOLO на последовательностях из {args.seq_dir}...")
    
    # Загружаем ground truth один раз
    gt_data = None
    if args.compute_metrics:
        gt_path = os.path.join(args.seq_dir, "gt.jsonl")
        gt_data = load_ground_truth(gt_path)
    
    # Обрабатываем каждую последовательность
    for seq_id in tqdm(valid_sequences, desc="Обработка последовательностей"):
        process_sequence(
            model, 
            args.seq_dir, 
            seq_id, 
            args.out_dir, 
            args.prefix, 
            compute_metrics=args.compute_metrics,
            gt_data=gt_data,
            conf_threshold=args.conf
        )


if __name__ == "__main__":
    main() 