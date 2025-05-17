#!/usr/bin/env python3
"""
Модуль для создания визуализаций классификаторов ResNet и VGG.
Содержит функции для визуализации точности и стабильности классификаторов
и создания GIF-анимаций в стиле легаси кода.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import imageio
import shutil
import cv2

# Импортируем общие утилиты
from scripts.utils import (
    load_image, find_sequence_frames, create_frames_for_gif, save_gif_animation,
    create_temp_directory, cleanup_temp_directory, get_boxplot_data, 
    load_csv_results
)


# =====================================================================
# Функции для работы с цветом
# =====================================================================

def get_color_by_value(value, min_value=75, max_value=100, metric="cos_sim"):
    """
    Возвращает цвет от красного к зеленому в зависимости от значения метрики.
    
    Args:
        value: значение метрики
        min_value: минимальное значение диапазона
        max_value: максимальное значение диапазона
        metric: тип метрики ("cos_sim" или "conf_drift")
        
    Returns:
        tuple: цвет в формате RGB (r, g, b) с значениями от 0 до 1
    """
    # Для метрики cosine similarity используем значение как есть
    # Для conf_drift - инвертируем (меньше = лучше)
    if metric == "conf_drift":
        normalized_value = 1.0 - ((value - min_value) / (max_value - min_value))
    else:
        normalized_value = (value - min_value) / (max_value - min_value)
    
    # Ограничиваем значение между 0 и 1
    normalized_value = max(0, min(1, normalized_value))
    
    # Используем градиент (красный -> желтый -> зеленый)
    if normalized_value < 0.5:
        # От красного к желтому
        r = 1.0
        g = normalized_value * 2
        b = 0.0
    else:
        # От желтого к зеленому
        r = 1.0 - 2 * (normalized_value - 0.5)
        g = 1.0
        b = 0.0
    
    # Убедимся, что значения в диапазоне [0, 1]
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))
    
    return (r, g, b)


# =====================================================================
# Функции для создания кадров в стиле легаси кода
# =====================================================================

def create_frame_with_legacy_style(
    img, frame_idx, total_frames, all_frames, models_data, models_current,
    metric="cos_sim", min_value=0.7, max_value=1.0,
    fixed_size=(16, 8), text_color="black", object_class="sparrow"
):
    """
    Создает кадр с изображением и боксплотами в стиле legacy кода
    
    Args:
        img: изображение для отображения
        frame_idx: индекс текущего кадра
        total_frames: общее количество кадров в последовательности
        all_frames: список всех номеров кадров
        models_data: словарь {model_name: all_values} для всех моделей
        models_current: словарь {model_name: current_value} для текущего кадра
        metric: метрика для визуализации ("cos_sim", "conf_drift" или "confidence")
        min_value: минимальное значение для оси Y
        max_value: максимальное значение для оси Y
        fixed_size: фиксированный размер фигуры (ширина, высота) в дюймах
        text_color: цвет текста для подписей
        object_class: класс объекта
        
    Returns:
        fig: фигура matplotlib
    """
    # Создаем фигуру с двумя подграфиками (изображение и боксплот)
    fig, (ax_img, ax_box) = plt.subplots(1, 2, figsize=fixed_size, 
                                      gridspec_kw={'width_ratios': [1, 1]})
    
    # Отображаем изображение
    ax_img.imshow(img)
    ax_img.axis('off')
    
    # Настраиваем подготовку данных в зависимости от метрики
    if metric == "cos_sim":
        # Преобразуем косинусное сходство [0-1] в проценты [0-100]
        display_min = min_value * 100
        display_max = max_value * 100
        display_values = {model: value * 100 for model, value in models_current.items()}
        display_data = {model: np.array(values) * 100 for model, values in models_data.items()}
        y_label = "Cosine Similarity (%)"
        title = f"Cosine Similarity for: {object_class}"
    elif metric == "conf_drift":
        # Преобразуем drift [0-1] в проценты [0-100]
        display_min = min_value * 100
        display_max = max_value * 100
        display_values = {model: value * 100 for model, value in models_current.items()}
        display_data = {model: np.array(values) * 100 for model, values in models_data.items()}
        y_label = "Confidence Drift (%)"
        title = f"Confidence Drift for: {object_class}"
    elif metric == "confidence":
        # Преобразуем confidence [0-1] в проценты [0-100]
        display_min = min_value * 100
        display_max = max_value * 100
        display_values = {model: value * 100 for model, value in models_current.items()}
        display_data = {model: np.array(values) * 100 for model, values in models_data.items()}
        y_label = "Confidence (%)"
        title = f"Confidence for: {object_class}"
    else:
        # Для других метрик
        display_min = min_value 
        display_max = max_value
        display_values = models_current
        display_data = models_data
        y_label = metric.replace('_', ' ').title()
        title = f"{y_label} for: {object_class}"
    
    # Настраиваем график боксплотов
    ax_box.set_ylim(display_min, display_max)
    
    # Позиции боксплотов и их метки
    model_names = list(models_data.keys())
    num_models = len(model_names)
    positions = list(range(1, num_models + 1))
    ax_box.set_xlim(0.5, num_models + 0.5)
    
    # Сокращаем длинные имена для компактности
    display_labels = []
    for name in model_names:
        if "ResNet" in name:
            if "AA-" in name:
                display_labels.append("Anti-aliased")
            elif "TIPS-" in name:
                display_labels.append("TIPS")
            else:
                display_labels.append("Baseline")
        elif "VGG" in name:
            if "AA-" in name:
                display_labels.append("Anti-aliased")
            elif "TIPS-" in name:
                display_labels.append("TIPS")
            else:
                display_labels.append("Baseline")
        else:
            display_labels.append(name)
    
    # Получаем статистики для боксплотов
    model_stats = {}
    for model in model_names:
        values = display_data[model]
        model_stats[model] = {
            'median': np.median(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'whislo': np.min(values),
            'whishi': np.max(values)
        }
    
    # Добавляем цветные бины и боксплоты
    for i, (model, pos) in enumerate(zip(model_names, positions)):
        stats = model_stats[model]
        value = display_values[model]
        
        # Получаем цвет бина в зависимости от значения
        color = get_color_by_value(value, display_min, display_max, metric)
        
        # 1. Сначала рисуем цветной бин от min_value до текущего значения
        bin_height = value - display_min
        bin_rect = Rectangle(
            (pos - 0.4, display_min), 
            0.8, 
            bin_height,
            facecolor=color, 
            edgecolor=None,
            alpha=0.8  # Непрозрачность как в legacy коде
        )
        ax_box.add_patch(bin_rect)
        
        # 2. Затем рисуем стандартный боксплот поверх
        # Бокс
        box_height = stats['q3'] - stats['q1']
        box = Rectangle(
            (pos - 0.4, stats['q1']), 
            0.8, 
            box_height, 
            facecolor='white', 
            edgecolor='black', 
            linewidth=1.5, 
            alpha=0.3  # Прозрачность как в legacy коде
        )
        ax_box.add_patch(box)
        
        # Медиана
        ax_box.hlines(stats['median'], pos - 0.4, pos + 0.4, colors='black', linewidth=2.0, alpha=0.5)
        
        # Усы
        ax_box.vlines(pos, stats['q1'], stats['whislo'], colors='black', linewidth=1.5, alpha=0.4)
        ax_box.vlines(pos, stats['q3'], stats['whishi'], colors='black', linewidth=1.5, alpha=0.4)
        ax_box.hlines(stats['whislo'], pos - 0.2, pos + 0.2, colors='black', linewidth=1.5, alpha=0.4)
        ax_box.hlines(stats['whishi'], pos - 0.2, pos + 0.2, colors='black', linewidth=1.5, alpha=0.4)
        
        # Подпись с текущим значением
        if metric == "cos_sim" or metric == "conf_drift" or metric == "confidence":
            value_label = f"{value:.1f}%"
        else:
            value_label = f"{value:.2f}"
            
        ax_box.text(pos, value + (display_max - display_min) * 0.02, value_label, 
                  ha='center', va='bottom', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Настройка осей и подписей
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(display_labels, fontsize=12)
    ax_box.set_ylabel(y_label, fontsize=14)
    ax_box.set_title(title, fontsize=16)
    ax_box.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Добавляем информацию о кадре
    # Показываем прогресс анимации
    if frame_idx in all_frames:
        current_frame_number = all_frames.index(frame_idx)
        progress = current_frame_number / max(1, (total_frames - 1))
        frame_label = f"Frame: {frame_idx} (Progress: {progress*100:.1f}%)"
    else:
        frame_label = f"Frame: {frame_idx}"
        
    fig.suptitle(frame_label, fontsize=14, y=0.95)
    
    plt.tight_layout()
    
    return fig


# =====================================================================
# Функции для создания GIF-анимаций
# =====================================================================

def create_classifier_comparison_gif(
    seq_dir, model_results, output_path=None, 
    sequence="seq_0", metric="cos_sim", fps=5, 
    start_frame=0, end_frame=None, step=1, slow_factor=3,
    min_value=0.7, max_value=1.0, object_class="sparrow"
):
    """
    Создает GIF-анимацию сравнения классификаторов в стиле legacy кода
    
    Args:
        seq_dir: директория с последовательностями изображений
        model_results: словарь {model_name: csv_path} с результатами моделей
        output_path: путь для сохранения GIF
        sequence: имя последовательности
        metric: метрика для визуализации ("cos_sim", "conf_drift" или "confidence")
        fps: частота кадров в GIF
        start_frame: начальный кадр
        end_frame: конечный кадр (None для всех кадров)
        step: шаг между кадрами
        slow_factor: фактор замедления (дублирование кадров)
        min_value: минимальное значение для оси Y
        max_value: максимальное значение для оси Y
        object_class: класс объекта
        
    Returns:
        str: путь к созданному GIF-файлу
    """
    # Проверка аргументов
    if not output_path:
        metric_name = metric.lower().replace(' ', '_')
        model_name = list(model_results.keys())[0].split('-')[0].lower()
        output_path = f"figures/boxplot_gifs/{sequence}_{model_name}_{metric_name}.gif"
    
    # Загружаем данные для каждой модели
    models_data = {}
    for model_name, csv_path in model_results.items():
        df = pd.read_csv(csv_path)
        if metric in df.columns:
            models_data[model_name] = df[metric].values
        else:
            print(f"Warning: Metric '{metric}' not found in {csv_path}")
            continue
    
    if not models_data:
        print("Error: No valid model data found")
        return None
    
    # Находим кадры последовательности
    frame_indices, frame_paths = find_sequence_frames(seq_dir, sequence, start_frame, end_frame, step)
    if not frame_indices:
        print(f"Error: No frames found for sequence '{sequence}'")
        return None
    
    # Создаем временную директорию для кадров
    tmp_dir = create_temp_directory(output_path)
    
    # Подготавливаем кадры GIF
    output_frame_paths = []
    
    print(f"Генерация {len(frame_indices)} кадров анимации...")
    
    for i, frame_idx in enumerate(frame_indices):
        if i % 10 == 0:
            print(f"Обработано {i} кадров...")
            
        # Загружаем изображение из найденного пути
        img_path = frame_paths[i]
        if not os.path.exists(img_path):
            print(f"Warning: Frame file not found: {img_path}")
            continue
        
        img = load_image(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}")
            continue
        
        # Получаем текущие значения метрик для каждой модели
        models_current = {}
        for model_name, values in models_data.items():
            if 0 <= frame_idx < len(values):
                models_current[model_name] = values[frame_idx]
            else:
                print(f"Warning: Frame {frame_idx} out of range for model {model_name}")
                models_current[model_name] = np.nan
        
        # Создаем кадр с боксплотами в стиле legacy
        fig = create_frame_with_legacy_style(
            img, frame_idx, len(frame_indices), frame_indices, models_data, models_current,
            metric=metric, min_value=min_value, max_value=max_value,
            fixed_size=(16, 8), text_color="black", object_class=object_class
        )
        
        # Сохраняем кадр
        frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        output_frame_paths.append(frame_path)
    
    # Создаем GIF с одинаковым размером для всех кадров
    try:
        frames_for_gif = []
        
        # Применяем фактор замедления
        if slow_factor > 1:
            repeated_frame_paths = []
            for frame_path in output_frame_paths:
                repeated_frame_paths.extend([frame_path] * slow_factor)
            output_frame_paths = repeated_frame_paths
            
        # Загружаем первый кадр, чтобы определить его размер
        if output_frame_paths:
            first_frame = cv2.imread(str(output_frame_paths[0]))
            target_shape = first_frame.shape[:2]
            
            # Загружаем и изменяем размер всех кадров
            for frame_path in output_frame_paths:
                # Загружаем кадр
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Изменяем размер, если необходимо
                if frame.shape[:2] != target_shape:
                    frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
                
                frames_for_gif.append(frame)
            
            # Сохраняем GIF
            if frames_for_gif:
                imageio.mimsave(output_path, frames_for_gif, fps=fps)
                print(f"GIF animation saved to: {output_path}")
            else:
                print("Error: No frames were processed for the GIF")
        else:
            print("Error: No frame files found")
    
    except Exception as e:
        import traceback
        print(f"Error creating GIF: {str(e)}")
        traceback.print_exc()
    
    # Очищаем временную директорию
    cleanup_temp_directory(tmp_dir)
    
    return output_path


# =====================================================================
# Пример использования
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create classifier comparison GIF")
    parser.add_argument("--seq_dir", type=str, required=True, help="Directory with sequences")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with classifier results")
    parser.add_argument("--out_path", type=str, help="Output GIF path")
    parser.add_argument("--sequence", type=str, default="seq_0", help="Sequence name")
    parser.add_argument("--metric", type=str, default="cos_sim", choices=["cos_sim", "conf_drift", "confidence"], 
                        help="Metric to visualize")
    parser.add_argument("--model_base", type=str, default="ResNet50", 
                        choices=["ResNet50", "VGG16"], help="Base model type")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame")
    parser.add_argument("--end_frame", type=int, default=None, help="Ending frame")
    parser.add_argument("--step", type=int, default=1, help="Frame step")
    parser.add_argument("--slow_factor", type=int, default=3, help="Slow factor")
    parser.add_argument("--min_value", type=float, default=None, 
                        help="Minimum value for y-axis (default: 0.7 for cos_sim, 0.0 for conf_drift, 0.7 for confidence)")
    parser.add_argument("--max_value", type=float, default=None, 
                        help="Maximum value for y-axis (default: 1.0 for cos_sim, 0.3 for conf_drift, 1.0 for confidence)")
    parser.add_argument("--object_class", type=str, default="sparrow", help="Object class name")
    
    args = parser.parse_args()
    
    # Устанавливаем значения min_value и max_value по умолчанию в зависимости от метрики
    if args.min_value is None:
        if args.metric == "cos_sim":
            args.min_value = 0.7
        elif args.metric == "conf_drift":
            args.min_value = 0.0
        elif args.metric == "confidence":
            args.min_value = 0.7
        else:
            args.min_value = 0.0

    if args.max_value is None:
        if args.metric == "cos_sim":
            args.max_value = 1.0
        elif args.metric == "conf_drift":
            args.max_value = 0.3
        elif args.metric == "confidence":
            args.max_value = 1.0
        else:
            args.max_value = 1.0
    
    # Формируем словарь с путями к данным моделей
    model_results = {
        f"{args.model_base}": os.path.join(args.results_dir, f"{args.model_base}_{args.sequence}.csv"),
        f"AA-{args.model_base}": os.path.join(args.results_dir, f"AA-{args.model_base}_{args.sequence}.csv"),
        f"TIPS-{args.model_base}": os.path.join(args.results_dir, f"TIPS-{args.model_base}_{args.sequence}.csv")
    }
    
    # Проверяем существование файлов результатов
    model_results = {k: v for k, v in model_results.items() if os.path.exists(v)}
    
    if not model_results:
        print("Error: No result files found")
        exit(1)
    
    # Создаем GIF-анимацию
    create_classifier_comparison_gif(
        args.seq_dir, model_results, args.out_path,
        sequence=args.sequence, metric=args.metric, fps=args.fps,
        start_frame=args.start_frame, end_frame=args.end_frame, step=args.step,
        slow_factor=args.slow_factor, min_value=args.min_value, max_value=args.max_value,
        object_class=args.object_class
    ) 