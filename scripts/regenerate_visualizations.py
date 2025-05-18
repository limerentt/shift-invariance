#!/usr/bin/env python3
"""
Скрипт для перегенерации всех визуализаций используя реальные данные.
Запускает модели на последовательностях, обновляет CSV с актуальными значениями
и регенерирует все визуализации.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Импортируем утилиты из других модулей проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import find_sequence_ids

# Импортируем зависимые модули только при необходимости, чтобы скрипт можно было
# запустить в среде без PyTorch для проверки и регенерации визуализаций


def run_command(command):
    """
    Запускает shell-команду и возвращает её вывод
    
    Args:
        command: команда для запуска
        
    Returns:
        str: вывод команды
    """
    print(f"Выполнение команды: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Ошибка при выполнении команды: {stderr.decode('utf-8')}")
        return None
    
    return stdout.decode('utf-8')


def run_models_on_sequences(args):
    """
    Запускает все модели на последовательностях изображений
    
    Args:
        args: аргументы командной строки
        
    Returns:
        bool: True, если успешно
    """
    # Директории
    seq_dir = args.seq_dir
    classifiers_results_dir = os.path.join(args.results_dir, "classifiers")
    yolo_results_dir = os.path.join(args.results_dir, "yolo")
    
    # Создаем директории для результатов, если их нет
    os.makedirs(classifiers_results_dir, exist_ok=True)
    os.makedirs(yolo_results_dir, exist_ok=True)
    
    # Команда для активации виртуальной среды
    venv_python = args.venv_python
    
    # Запускаем классификаторы
    print("Запуск классификаторов на последовательностях...")
    classifier_cmd = (
        f"{venv_python} -m scripts.run_classifiers "
        f"--seq-dir {seq_dir} "
        f"--out-dir {classifiers_results_dir} "
        f"--models vgg16,aa-vgg16,resnet50,aa-resnet50,tips-vgg16,tips-resnet50 "
        f"--device {args.device}"
    )
    
    run_command(classifier_cmd)
    
    # Запускаем детекторы YOLO
    print("Запуск детекторов YOLO на последовательностях...")
    
    # Базовая YOLO модель
    detector_cmd = (
        f"{venv_python} -m scripts.run_detector "
        f"--weights checkpoints/yolov5s.pt "
        f"--seq-dir {seq_dir} "
        f"--out-dir {yolo_results_dir} "
        f"--prefix baseline "
        f"--conf 0.25"
    )
    
    run_command(detector_cmd)
    
    # TIPS-YOLO модель
    tips_detector_cmd = (
        f"{venv_python} -m scripts.run_detector "
        f"--weights results/yolo_tips/yolo_tips_model.pt "
        f"--seq-dir {seq_dir} "
        f"--out-dir {yolo_results_dir} "
        f"--prefix tips "
        f"--conf 0.25"
    )
    
    run_command(tips_detector_cmd)
    
    # Обновляем значения confidence в CSV-файлах, используя реальные данные
    print("Обновление метрик в CSV-файлах с использованием реальных данных...")
    update_cmd = (
        f"{venv_python} -m scripts.update_classifier_metrics "
        f"--seq-dir {seq_dir} "
        f"--results-dir {classifiers_results_dir} "
        f"--model-output-dir {classifiers_results_dir}/outputs "
        f"--model-type all"
    )
    
    run_command(update_cmd)
    
    return True


def check_csv_data_integrity(results_dir):
    """
    Проверяет целостность данных в CSV-файлах
    
    Args:
        results_dir: директория с результатами
        
    Returns:
        tuple: (is_valid, issues) - флаг валидности и список проблем
    """
    issues = []
    classifiers_dir = os.path.join(results_dir, "classifiers")
    yolo_dir = os.path.join(results_dir, "yolo")
    
    # Проверяем наличие директорий
    if not os.path.exists(classifiers_dir):
        issues.append(f"Директория результатов классификаторов не найдена: {classifiers_dir}")
    
    if not os.path.exists(yolo_dir):
        issues.append(f"Директория результатов YOLO не найдена: {yolo_dir}")
    
    if issues:
        return False, issues
    
    # Проверяем файлы классификаторов
    classifier_models = ["VGG16", "AA-VGG16", "TIPS-VGG16", "ResNet50", "AA-ResNet50", "TIPS-ResNet50"]
    sequences = ["seq_0", "seq_1", "seq_2"]
    
    for model in classifier_models:
        for seq in sequences:
            csv_path = os.path.join(classifiers_dir, f"{model}_{seq}.csv")
            if not os.path.exists(csv_path):
                issues.append(f"Отсутствует файл результатов классификатора: {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                if 'confidence' not in df.columns:
                    issues.append(f"В файле {csv_path} отсутствует столбец confidence")
                elif df['confidence'].isna().any() or (df['confidence'] == 0).all():
                    issues.append(f"В файле {csv_path} некорректные значения confidence")
            except Exception as e:
                issues.append(f"Ошибка при чтении файла {csv_path}: {e}")
    
    # Проверяем файлы YOLO
    yolo_models = ["baseline", "yolo", "tips"]
    
    for model in yolo_models:
        for seq in sequences:
            csv_path = os.path.join(yolo_dir, f"{model}_{seq}.csv")
            if not os.path.exists(csv_path):
                issues.append(f"Отсутствует файл результатов YOLO: {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                if 'confidence' not in df.columns:
                    issues.append(f"В файле {csv_path} отсутствует столбец confidence")
                elif df['confidence'].isna().any() or (df['confidence'] == 0).all():
                    issues.append(f"В файле {csv_path} некорректные значения confidence")
            except Exception as e:
                issues.append(f"Ошибка при чтении файла {csv_path}: {e}")
    
    return len(issues) == 0, issues


def regenerate_all_visualizations(args):
    """
    Перегенерирует все визуализации на основе обновленных данных
    
    Args:
        args: аргументы командной строки
        
    Returns:
        bool: True, если успешно
    """
    # Команда для активации виртуальной среды
    venv_python = args.venv_python
    
    # Запускаем скрипт для создания визуализаций
    visualizations_cmd = f"{venv_python} -m scripts.create_visualizations"
    run_command(visualizations_cmd)
    
    # Комментируем генерацию GIF-анимаций, так как это занимает много времени
    """
    # Генерируем YOLO GIF анимации
    yolo_gif_cmd = (
        f"{venv_python} -m scripts.visualization_yolo "
        f"--seq-dir {args.seq_dir} "
        f"--results-dir {os.path.join(args.results_dir, 'yolo')} "
        f"--output-dir {os.path.join(args.figures_dir, 'yolo_gifs')} "
        f"--mode gif --fps 5"
    )
    run_command(yolo_gif_cmd)
    
    # Генерируем боксплоты и другие анимации для ResNet50
    for model_base in ["ResNet50", "VGG16"]:
        for metric in ["cos_sim", "conf_drift", "confidence"]:
            for seq in ["seq_0", "seq_1", "seq_2"]:
                classifier_gif_cmd = (
                    f"{venv_python} -m scripts.visualization_classifiers "
                    f"--seq_dir {args.seq_dir} "
                    f"--results_dir {os.path.join(args.results_dir, 'classifiers')} "
                    f"--out_path {os.path.join(args.figures_dir, 'boxplot_gifs', f'{model_base}_{metric}_{seq}.gif')} "
                    f"--sequence {seq} "
                    f"--metric {metric} "
                    f"--model_base {model_base} "
                    f"--fps 5"
                )
                run_command(classifier_gif_cmd)
    """
    
    return True


def copy_figures_to_latex(args):
    """
    Копирует сгенерированные изображения в директорию LaTeX проекта
    
    Args:
        args: аргументы командной строки
        
    Returns:
        bool: True, если успешно
    """
    latex_images_dir = os.path.join(args.latex_dir, "images")
    
    # Создаем директорию, если её нет
    os.makedirs(latex_images_dir, exist_ok=True)
    
    # Копируем все изображения из директории figures в latex/images
    copy_cmd = f"cp -r {args.figures_dir}/* {latex_images_dir}"
    run_command(copy_cmd)
    
    print(f"Изображения скопированы в директорию LaTeX: {latex_images_dir}")
    return True


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Перегенерация всех визуализаций с использованием реальных данных"
    )
    
    # Обязательные аргументы
    parser.add_argument(
        "--seq-dir", type=str, default="../data/sequences",
        help="Директория с последовательностями изображений"
    )
    parser.add_argument(
        "--results-dir", type=str, default="../results",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--figures-dir", type=str, default="../figures",
        help="Директория для сохранения визуализаций"
    )
    parser.add_argument(
        "--latex-dir", type=str, default="../diploma_pdf/Dissertation",
        help="Директория с LaTeX проектом диплома"
    )
    
    # Опциональные аргументы
    parser.add_argument(
        "--skip-models", action="store_true",
        help="Пропустить запуск моделей (использовать существующие результаты)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Принудительно продолжить даже при обнаружении проблем"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Устройство для запуска моделей (cuda/cpu)"
    )
    parser.add_argument(
        "--venv-python", type=str, default="python",
        help="Путь к исполняемому файлу Python в виртуальном окружении"
    )
    
    return parser.parse_args()


def main():
    """Основная функция"""
    args = parse_args()
    
    # Фиксируем seed для воспроизводимости
    np.random.seed(42)
    
    # Запускаем модели, если не указан флаг --skip-models
    if not args.skip_models:
        print("Запуск моделей на последовательностях...")
        run_models_on_sequences(args)
    else:
        print("Запуск моделей пропущен (использование существующих результатов)")
    
    # Проверяем целостность данных в CSV-файлах
    print("Проверка целостности данных...")
    is_valid, issues = check_csv_data_integrity(args.results_dir)
    
    if not is_valid:
        print("Обнаружены проблемы с данными:")
        for issue in issues:
            print(f"  - {issue}")
        
        if not args.force:
            print("Ошибка: Обнаружены проблемы с данными. Используйте --force для принудительного продолжения.")
            return False
        
        print("Предупреждение: Обнаружены проблемы с данными, но продолжаем из-за флага --force")
    
    # Перегенерируем все визуализации
    print("Перегенерация всех визуализаций...")
    regenerate_all_visualizations(args)
    
    # Копируем изображения в директорию LaTeX
    print("Копирование изображений в директорию LaTeX...")
    copy_figures_to_latex(args)
    
    print("\nГотово! Все визуализации успешно перегенерированы.")
    return True


if __name__ == "__main__":
    main() 