#!/usr/bin/env python3
"""
Интерфейс командной строки для создания различных визуализаций
и GIF-анимаций с использованием модульной структуры проекта.
"""

import os
import sys
import argparse
from pathlib import Path

# Добавляем родительскую директорию в sys.path для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import find_sequence_ids
from scripts.visualization import create_comparison_gif
from scripts.visualization_classifiers import create_classifier_comparison_gif
from scripts.visualization_yolo import create_yolo_comparison_gif
# Больше не импортируем специфичные функции для YOLO, используем обобщенную функцию


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Создание различных визуализаций и GIF-анимаций моделей"
    )
    
    # Основные подкоманды
    subparsers = parser.add_subparsers(
        dest="command",
        help="Подкоманда для выполнения"
    )
    
    # === Команда для создания GIF сравнения моделей детекции ===
    gif_parser = subparsers.add_parser(
        "comparison-gif",
        help="Создание GIF-анимации сравнения моделей детекции с боксплотами"
    )
    
    # Обязательные аргументы
    gif_parser.add_argument(
        "--seq-dir", type=str, required=True, 
        help="Директория с последовательностями изображений"
    )
    gif_parser.add_argument(
        "--baseline", type=str, required=True, 
        help="Путь к CSV с результатами базовой модели"
    )
    gif_parser.add_argument(
        "--out-dir", type=str, required=True, 
        help="Директория для сохранения GIF-анимаций"
    )
    
    # Опциональные аргументы
    gif_parser.add_argument(
        "--modified", type=str, default=None,
        help="Путь к CSV с результатами улучшенной модели (например, TIPS-модель)"
    )
    gif_parser.add_argument(
        "--sequence", type=str, default=None,
        help="ID конкретной последовательности для обработки (если None, обрабатываются все)"
    )
    gif_parser.add_argument(
        "--fps", type=int, default=5,
        help="Частота кадров GIF-анимации"
    )
    gif_parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Начальный кадр последовательности"
    )
    gif_parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Конечный кадр последовательности (None = все)"
    )
    gif_parser.add_argument(
        "--step", type=int, default=1,
        help="Шаг между кадрами"
    )
    gif_parser.add_argument(
        "--slow-factor", type=int, default=3,
        help="Коэффициент замедления анимации"
    )
    gif_parser.add_argument(
        "--width", type=int, default=1200,
        help="Ширина кадров GIF в пикселях"
    )
    gif_parser.add_argument(
        "--height", type=int, default=800,
        help="Высота кадров GIF в пикселях"
    )
    gif_parser.add_argument(
        "--min-conf", type=float, default=70,
        help="Минимальное значение для оси Y боксплота (в %)"
    )
    gif_parser.add_argument(
        "--max-conf", type=float, default=100,
        help="Максимальное значение для оси Y боксплота (в %)"
    )
    gif_parser.add_argument(
        "--bg-color", type=str, default="white", choices=["white", "black"],
        help="Цвет фона для визуализации"
    )
    gif_parser.add_argument(
        "--enhancement", type=str, default="moderate", 
        choices=["none", "light", "moderate", "strong"],
        help="Уровень улучшения качества изображения"
    )
    gif_parser.add_argument(
        "--object-class", type=str, default=None,
        help="Класс объекта для отображения в заголовке (автоопределение если None)"
    )
    gif_parser.add_argument(
        "--baseline-label", type=str, default="Baseline",
        help="Метка для базовой модели"
    )
    gif_parser.add_argument(
        "--modified-label", type=str, default="TIPS",
        help="Метка для улучшенной модели"
    )
    
    # === Команда для создания GIF классификаторов ===
    classifier_gif_parser = subparsers.add_parser(
        "classifier-gif",
        help="Создание GIF-анимации сравнения классификаторов"
    )
    
    # Обязательные аргументы
    classifier_gif_parser.add_argument(
        "--seq-dir", type=str, required=True, 
        help="Директория с последовательностями изображений"
    )
    classifier_gif_parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Директория с результатами классификаторов"
    )
    classifier_gif_parser.add_argument(
        "--out-dir", type=str, required=True, 
        help="Директория для сохранения GIF-анимаций"
    )
    
    # Опциональные аргументы
    classifier_gif_parser.add_argument(
        "--sequence", type=str, default=None,
        help="ID конкретной последовательности для обработки (если None, обрабатываются все)"
    )
    classifier_gif_parser.add_argument(
        "--model-type", type=str, choices=["resnet", "vgg", "all"], default="all", 
        help="Тип моделей для визуализации"
    )
    classifier_gif_parser.add_argument(
        "--metric", type=str, choices=["cos_sim", "conf_drift"], default="cos_sim", 
        help="Метрика для визуализации"
    )
    classifier_gif_parser.add_argument(
        "--fps", type=int, default=5,
        help="Частота кадров GIF-анимации"
    )
    classifier_gif_parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Начальный кадр последовательности"
    )
    classifier_gif_parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Конечный кадр последовательности (None = все)"
    )
    classifier_gif_parser.add_argument(
        "--step", type=int, default=1,
        help="Шаг между кадрами"
    )
    classifier_gif_parser.add_argument(
        "--width", type=int, default=1200,
        help="Ширина кадров GIF в пикселях"
    )
    classifier_gif_parser.add_argument(
        "--height", type=int, default=800,
        help="Высота кадров GIF в пикселях"
    )
    classifier_gif_parser.add_argument(
        "--bg-color", type=str, default="white", choices=["white", "black"],
        help="Цвет фона для визуализации"
    )
    classifier_gif_parser.add_argument(
        "--enhancement", type=str, default="moderate", 
        choices=["none", "light", "moderate", "strong"],
        help="Уровень улучшения качества изображения"
    )
    classifier_gif_parser.add_argument(
        "--object-class", type=str, default="sparrow",
        help="Класс объекта для отображения в заголовке"
    )
    
    # Добавляем аргументы, специфичные для метрики cos_sim
    classifier_gif_parser.add_argument(
        "--min-cos-sim", type=float, default=0.7,
        help="Минимальное значение для оси Y при отображении косинусного сходства"
    )
    classifier_gif_parser.add_argument(
        "--max-cos-sim", type=float, default=1.0,
        help="Максимальное значение для оси Y при отображении косинусного сходства"
    )
    
    # Добавляем аргументы, специфичные для метрики conf_drift
    classifier_gif_parser.add_argument(
        "--min-conf-drift", type=float, default=0.0,
        help="Минимальное значение для оси Y при отображении дрейфа уверенности"
    )
    classifier_gif_parser.add_argument(
        "--max-conf-drift", type=float, default=0.1,
        help="Максимальное значение для оси Y при отображении дрейфа уверенности"
    )
    
    # === Команда для создания GIF сравнения YOLO моделей ===
    yolo_gif_parser = subparsers.add_parser(
        "yolo-comparison",
        help="Создание GIF-анимации сравнения моделей YOLO с боксплотами уверенности"
    )
    
    # Обязательные аргументы
    yolo_gif_parser.add_argument(
        "--seq-dir", type=str, required=True, 
        help="Директория с последовательностями изображений"
    )
    yolo_gif_parser.add_argument(
        "--baseline", type=str, required=True, 
        help="Путь к CSV с результатами базовой YOLO модели"
    )
    yolo_gif_parser.add_argument(
        "--out-dir", type=str, required=True, 
        help="Директория для сохранения GIF-анимаций"
    )
    
    # Опциональные аргументы
    yolo_gif_parser.add_argument(
        "--modified", type=str, default=None,
        help="Путь к CSV с результатами TIPS-YOLO модели"
    )
    yolo_gif_parser.add_argument(
        "--sequence", type=str, default=None,
        help="ID конкретной последовательности для обработки (если None, обрабатываются все)"
    )
    yolo_gif_parser.add_argument(
        "--fps", type=int, default=5,
        help="Частота кадров GIF-анимации"
    )
    yolo_gif_parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Начальный кадр последовательности"
    )
    yolo_gif_parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Конечный кадр последовательности (None = все)"
    )
    yolo_gif_parser.add_argument(
        "--step", type=int, default=1,
        help="Шаг между кадрами"
    )
    yolo_gif_parser.add_argument(
        "--slow-factor", type=int, default=3,
        help="Коэффициент замедления анимации"
    )
    yolo_gif_parser.add_argument(
        "--width", type=int, default=16,
        help="Ширина фигуры в дюймах"
    )
    yolo_gif_parser.add_argument(
        "--height", type=int, default=8,
        help="Высота фигуры в дюймах"
    )
    yolo_gif_parser.add_argument(
        "--min-conf", type=float, default=75.0,
        help="Минимальное значение для оси Y боксплота (в %%)"
    )
    yolo_gif_parser.add_argument(
        "--max-conf", type=float, default=85.0,
        help="Максимальное значение для оси Y боксплота (в %%)"
    )
    yolo_gif_parser.add_argument(
        "--object-class", type=str, default="object",
        help="Класс объекта для отображения в заголовке"
    )
    yolo_gif_parser.add_argument(
        "--output-name", type=str, default=None,
        help="Имя выходного файла (без пути)"
    )
    
    return parser.parse_args()


def create_comparison_gifs(args):
    """
    Создает GIF-анимации сравнения моделей детекции
    
    Args:
        args: аргументы командной строки
    """
    # Подготавливаем пути
    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Проверяем и корректируем размеры
    width = args.width
    height = args.height
    
    # Проверяем соотношение сторон (должно быть разумным, обычно ширина > высоты)
    aspect_ratio = width / height
    if aspect_ratio < 1.0 or aspect_ratio > 2.5:
        print(f"Предупреждение: Необычное соотношение сторон {aspect_ratio:.2f}. Устанавливаем стандартное 16:9")
        width = 1600
        height = 900
    
    fixed_size = (width, height)
    
    # Настройки контрастности
    min_conf = args.min_conf
    max_conf = args.max_conf
    
    print(f"Настройки визуализации: размер={fixed_size}, контрастность={min_conf}-{max_conf}, "
          f"фон={args.bg_color}, улучшение={args.enhancement}, FPS={args.fps}")
    
    # Определяем последовательности для обработки
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = find_sequence_ids(seq_dir)
        if not sequences:
            print(f"Ошибка: Не найдены последовательности в {seq_dir}")
            return
    
    print(f"Будут обработаны последовательности: {sequences}")
    
    # Для каждой последовательности создаем GIF-анимацию
    for sequence in sequences:
        print(f"\n{'-'*50}")
        print(f"Обработка последовательности: {sequence}")
        print(f"{'-'*50}")
        
        # Определяем пути
        output_path = out_dir / f"{sequence}_comparison.gif"
        
        # Формируем пути к CSV файлам с результатами
        if os.path.isdir(args.baseline):
            # Если указана директория, ищем файл с именем последовательности
            baseline_path = Path(args.baseline) / f"{sequence}.csv"
            if not baseline_path.exists():
                # Пробуем с префиксом baseline_
                baseline_path = Path(args.baseline) / f"baseline_{sequence}.csv"
        else:
            # Иначе используем путь как есть
            baseline_path = Path(args.baseline)
        
        if not baseline_path.exists():
            print(f"Ошибка: Не найден файл с baseline результатами по пути {baseline_path}")
            continue
        
        modified_path = None
        if args.modified:
            if os.path.isdir(args.modified):
                modified_path = Path(args.modified) / f"{sequence}.csv"
                if not modified_path.exists():
                    # Пробуем с префиксами tips_, modified_
                    for prefix in ["tips_", "modified_"]:
                        test_path = Path(args.modified) / f"{prefix}{sequence}.csv"
                        if test_path.exists():
                            modified_path = test_path
                            break
            else:
                modified_path = Path(args.modified)
            
            if not modified_path.exists():
                print(f"Предупреждение: Не найден файл с modified результатами по пути {modified_path}")
                modified_path = None
        
        # Создаем GIF-анимацию сравнения
        create_comparison_gif(
            seq_dir=str(seq_dir),
            baseline_results=str(baseline_path),
            modified_results=str(modified_path) if modified_path else None,
            output_path=str(output_path),
            sequence=sequence,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            slow_factor=args.slow_factor,
            fixed_size=fixed_size,
            min_conf=min_conf,
            max_conf=max_conf,
            bg_color=args.bg_color,
            enhancement_level=args.enhancement,
            object_class=args.object_class,
            baseline_label=args.baseline_label,
            modified_label=args.modified_label
        )
        
        print(f"Создана GIF-анимация: {output_path}")


def create_classifier_gifs(args):
    """Создание GIF-анимаций сравнения классификаторов"""
    # Подготавливаем пути
    seq_dir = Path(args.seq_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Проверяем и корректируем размеры
    width = args.width
    height = args.height
    
    # Проверяем соотношение сторон
    aspect_ratio = width / height
    if aspect_ratio < 1.0 or aspect_ratio > 2.5:
        print(f"Предупреждение: Необычное соотношение сторон {aspect_ratio:.2f}. Устанавливаем стандартное 1:1")
        width = 1200
        height = 800
    
    fixed_size = (width, height)
    
    # Настройки диапазона значений в зависимости от метрики
    if args.metric == "cos_sim":
        min_value = args.min_cos_sim
        max_value = args.max_cos_sim
    else:  # conf_drift
        min_value = args.min_conf_drift
        max_value = args.max_conf_drift
    
    print(f"Настройки визуализации: размер={fixed_size}, "
          f"метрика={args.metric}, диапазон={min_value}-{max_value}, "
          f"фон={args.bg_color}, улучшение={args.enhancement}, FPS={args.fps}")
    
    # Определяем последовательности для обработки
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = find_sequence_ids(seq_dir)
        if not sequences:
            print(f"Ошибка: Не найдены последовательности в {seq_dir}")
            return
    
    print(f"Будут обработаны последовательности: {sequences}")
    
    # Для каждой последовательности создаем GIF-анимации
    for sequence in sequences:
        print(f"\n{'-'*50}")
        print(f"Обработка последовательности: {sequence}")
        print(f"{'-'*50}")
        
        # Определяем модели в зависимости от выбранного типа
        all_models = {}
        
        if args.model_type == "resnet" or args.model_type == "all":
            print("Поиск моделей ResNet...")
            for model in ["ResNet50", "AA-ResNet50", "TIPS-ResNet50"]:
                csv_path = results_dir / f"{model}_{sequence}.csv"
                if csv_path.exists():
                    all_models[model] = str(csv_path)
                    print(f"  Найдена модель {model}")
        
        if args.model_type == "vgg" or args.model_type == "all":
            print("Поиск моделей VGG...")
            for model in ["VGG16", "AA-VGG16", "TIPS-VGG16"]:
                csv_path = results_dir / f"{model}_{sequence}.csv"
                if csv_path.exists():
                    all_models[model] = str(csv_path)
                    print(f"  Найдена модель {model}")
        
        if not all_models:
            print(f"Ошибка: Не найдены CSV-файлы для {args.model_type} моделей в директории {results_dir}")
            continue
        
        # Формируем выходной путь
        if args.model_type == "resnet":
            output_name = f"{sequence}_resnet_{args.metric}.gif"
        elif args.model_type == "vgg":
            output_name = f"{sequence}_vgg_{args.metric}.gif"
        else:
            output_name = f"{sequence}_all_{args.metric}.gif"
        
        output_path = out_dir / output_name
        
        # Создаем GIF-анимацию для классификаторов
        create_classifier_comparison_gif(
            seq_dir=str(seq_dir),
            model_results=all_models,
            output_path=str(output_path),
            sequence=sequence,
            metric=args.metric,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            min_value=min_value,
            max_value=max_value,
            object_class=args.object_class
        )
        
        print(f"Создана GIF-анимация: {output_path}")


def handle_yolo_comparison_command(args):
    """Обработка команды создания GIF-анимации сравнения YOLO моделей"""
    print(f"Настройки визуализации YOLO: размер=({args.width}, {args.height}), "
          f"диапазон конфиденса={args.min_conf}%-{args.max_conf}%, "
          f"объект={args.object_class}, FPS={args.fps}")
    
    # Проверяем, существует ли директория для сохранения
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Определяем последовательности для обработки
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = find_sequence_ids(args.seq_dir)
    
    print(f"Будут обработаны последовательности: {sequences}")
    
    # Обрабатываем каждую последовательность
    for sequence in sequences:
        print("\n" + "-" * 50)
        print(f"Обработка последовательности: {sequence}")
        print("-" * 50)
        
        # Определяем пути к файлам результатов
        if os.path.isdir(args.baseline):
            base_seq_file = f"{args.baseline}/{sequence}.csv"
            if not os.path.exists(base_seq_file):
                # Пробуем альтернативный формат имени
                base_seq_file = f"{args.baseline}/baseline_{sequence}.csv"
                if not os.path.exists(base_seq_file):
                    print(f"Пропуск: Не найден файл с результатами для {sequence}")
                    continue
        else:
            base_seq_file = args.baseline
        
        # Определяем путь для сохранения GIF
        if args.output_name:
            # Используем указанное имя файла
            output_gif = os.path.join(args.out_dir, args.output_name)
        else:
            # Генерируем имя автоматически
            if args.modified:
                output_gif = os.path.join(args.out_dir, f"{sequence}_yolo_comparison.gif")
            else:
                output_gif = os.path.join(args.out_dir, f"{sequence}_yolo_baseline.gif")
        
        # Определяем путь к результатам TIPS модели, если указан
        tips_file = None
        if args.modified:
            if os.path.isdir(args.modified):
                tips_file = f"{args.modified}/tips_{sequence}.csv"
                if not os.path.exists(tips_file):
                    print(f"Предупреждение: Не найден файл TIPS для {sequence}, "
                          f"будет показана только базовая модель")
                    tips_file = None
            else:
                tips_file = args.modified
        
        # Создаем GIF-анимацию
        create_yolo_comparison_gif(
            seq_dir=args.seq_dir,
            baseline_results=base_seq_file,
            tips_results=tips_file,
            output_path=output_gif,
            sequence=sequence,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            slow_factor=args.slow_factor,
            fixed_size=(args.width, args.height),
            min_conf=args.min_conf,
            max_conf=args.max_conf,
            object_class=args.object_class
        )
        
        print(f"Создана YOLO GIF-анимация: {output_gif}")


def main():
    """Основная функция"""
    args = parse_args()
    
    if args.command == "comparison-gif":
        create_comparison_gifs(args)
    elif args.command == "classifier-gif":
        create_classifier_gifs(args)
    elif args.command == "yolo-comparison":
        handle_yolo_comparison_command(args)
    else:
        print("Пожалуйста, укажите команду.")
        sys.exit(1)


if __name__ == "__main__":
    main() 