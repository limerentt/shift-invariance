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
# Импортируем новую функцию из модуля visualization_yolo
from scripts.visualization_yolo import create_yolo_comparison_gif


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Создание различных визуализаций и GIF-анимаций YOLO и других моделей"
    )
    
    # Основные подкоманды
    subparsers = parser.add_subparsers(
        dest="command",
        help="Подкоманда для выполнения"
    )
    
    # === Команда для создания GIF сравнения YOLO моделей ===
    yolo_parser = subparsers.add_parser(
        "yolo-comparison",
        help="Создание GIF-анимации сравнения YOLO моделей с боксплотами"
    )
    
    # Обязательные аргументы
    yolo_parser.add_argument(
        "--seq-dir", type=str, required=True, 
        help="Директория с последовательностями изображений"
    )
    yolo_parser.add_argument(
        "--baseline", type=str, required=True, 
        help="Путь к CSV с результатами базовой YOLO модели"
    )
    yolo_parser.add_argument(
        "--out-dir", type=str, required=True, 
        help="Директория для сохранения GIF-анимаций"
    )
    
    # Опциональные аргументы
    yolo_parser.add_argument(
        "--modified", type=str, default=None,
        help="Путь к CSV с результатами улучшенной YOLO модели (TIPS-YOLO)"
    )
    yolo_parser.add_argument(
        "--sequence", type=str, default=None,
        help="ID конкретной последовательности для обработки (если None, обрабатываются все)"
    )
    yolo_parser.add_argument(
        "--fps", type=int, default=5,
        help="Частота кадров GIF-анимации"
    )
    yolo_parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Начальный кадр последовательности"
    )
    yolo_parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Конечный кадр последовательности (None = все)"
    )
    yolo_parser.add_argument(
        "--step", type=int, default=1,
        help="Шаг между кадрами"
    )
    yolo_parser.add_argument(
        "--slow-factor", type=int, default=3,
        help="Коэффициент замедления анимации"
    )
    yolo_parser.add_argument(
        "--width", type=int, default=1200,
        help="Ширина кадров GIF в пикселях"
    )
    yolo_parser.add_argument(
        "--height", type=int, default=800,
        help="Высота кадров GIF в пикселях"
    )
    yolo_parser.add_argument(
        "--min-conf", type=float, default=75,
        help="Минимальное значение для оси Y боксплота (в %)"
    )
    yolo_parser.add_argument(
        "--max-conf", type=float, default=85,
        help="Максимальное значение для оси Y боксплота (в %)"
    )
    
    # === Можно добавить и другие подкоманды ===
    # Например, для боксплотов классификаторов, тепловых карт и т.д.
    
    return parser.parse_args()


def create_yolo_comparison_gifs(args):
    """
    Создает GIF-анимации сравнения YOLO моделей
    
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
    
    print(f"Настройки визуализации: размер={fixed_size}, контрастность={min_conf}-{max_conf}, FPS={args.fps}")
    
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
        
        tips_path = None
        if args.modified:
            if os.path.isdir(args.modified):
                tips_path = Path(args.modified) / f"{sequence}.csv"
                if not tips_path.exists():
                    # Пробуем с префиксом tips_ или modified_
                    for prefix in ["tips_", "modified_"]:
                        test_path = Path(args.modified) / f"{prefix}{sequence}.csv"
                        if test_path.exists():
                            tips_path = test_path
                            break
            else:
                tips_path = Path(args.modified)
            
            if tips_path and not tips_path.exists():
                print(f"Предупреждение: Не найден файл с TIPS результатами по пути {tips_path}")
                tips_path = None
        
        print(f"Baseline CSV: {baseline_path}")
        print(f"TIPS CSV: {tips_path}")
        
        # Создаем GIF-анимацию используя новую функцию из visualization_yolo
        result = create_yolo_comparison_gif(
            seq_dir=seq_dir,
            baseline_results=baseline_path,
            tips_results=tips_path,
            output_path=output_path,
            sequence=sequence,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            slow_factor=args.slow_factor,
            fixed_size=fixed_size,
            min_conf=min_conf,
            max_conf=max_conf
        )
        
        if result:
            print(f"✅ Создана GIF-анимация: {output_path}")
        else:
            print(f"❌ Ошибка при создании GIF-анимации для {sequence}")
            
    print("\n✅ Все процессы завершены.")


def main():
    """Основная функция"""
    args = parse_args()
    
    # Обработка команд
    if args.command == "yolo-comparison":
        create_yolo_comparison_gifs(args)
    else:
        print("Пожалуйста, укажите подкоманду. Используйте --help для получения справки.")


if __name__ == "__main__":
    main() 