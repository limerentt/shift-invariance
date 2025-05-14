#!/usr/bin/env python3
"""
Создает GIF-анимацию, показывающую сравнение трех моделей (baseline, BlurPool, TIPS)
в виде боксплотов с цветными бинами (от красного к зеленому) под ними, высота которых 
соответствует уверенности модели.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
from pathlib import Path
import argparse
import cv2
import os
import shutil
from scipy.interpolate import interp1d

def parse_args():
    parser = argparse.ArgumentParser(description="Create boxplot confidence GIF animation")
    parser.add_argument("--seq_dir", type=str, required=True, help="Directory with sequence images")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with classifier results")
    parser.add_argument("--out_path", type=str, required=True, help="Output GIF file path")
    parser.add_argument("--sequence", type=str, default="seq_0", help="Sequence to visualize")
    parser.add_argument("--model_base", type=str, default="ResNet50", help="Base model architecture")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second in the GIF")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame number")
    parser.add_argument("--end_frame", type=int, default=None, help="Ending frame number (None for all)")
    parser.add_argument("--step", type=int, default=1, help="Frame step size")
    parser.add_argument("--loop", type=int, default=0, help="Number of times to loop the GIF (0 = infinite)")
    parser.add_argument("--y_min", type=float, default=75, help="Minimum value for y-axis")
    parser.add_argument("--smooth", action="store_true", help="Create smoother animation")
    parser.add_argument("--slow_factor", type=int, default=4, help="Factor to slow down animation by repeating frames")
    return parser.parse_args()

def get_color(confidence, min_conf=75, max_conf=100):
    """
    Возвращает цвет в зависимости от уверенности.
    Красный для минимальных значений, зеленый для максимальных.
    """
    # Нормализуем значение confidence к диапазону [0, 1]
    normalized_conf = (confidence - min_conf) / (max_conf - min_conf)
    normalized_conf = max(0, min(1, normalized_conf))  # Ограничиваем значение между 0 и 1
    
    if normalized_conf < 0.5:
        # От красного к желтому
        r = 1.0
        g = normalized_conf * 2
        b = 0.0
    else:
        # От желтого к зеленому
        r = 1.0 - (normalized_conf - 0.5) * 2
        g = 1.0
        b = 0.0
    
    # Ограничиваем значения между 0 и 1
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    
    return (r, g, b)

def create_boxplot_confidence_gif(seq_dir, results_dir, sequence, model_base, out_path, fps, 
                                 start_frame=0, end_frame=None, step=1, loop=0, y_min=75,
                                 smooth=False, slow_factor=4):
    """Создает GIF-анимацию с боксплотами уверенности для трех моделей."""
    seq_dir = Path(seq_dir)
    results_dir = Path(results_dir)
    
    # Загружаем данные для трех моделей
    baseline_csv = results_dir / f"{model_base}_{sequence}.csv"
    blurpool_csv = results_dir / f"AA-{model_base}_{sequence}.csv"
    tips_csv = results_dir / f"TIPS-{model_base}_{sequence}.csv"
    
    if not all(f.exists() for f in [baseline_csv, blurpool_csv, tips_csv]):
        print(f"Error: Not all required CSV files exist in {results_dir}")
        return
    
    baseline_data = pd.read_csv(baseline_csv)
    blurpool_data = pd.read_csv(blurpool_csv)
    tips_data = pd.read_csv(tips_csv)
    
    # Преобразуем cos_sim в confidence (0-100%)
    baseline_data['confidence'] = baseline_data['cos_sim'] * 100
    blurpool_data['confidence'] = blurpool_data['cos_sim'] * 100
    tips_data['confidence'] = tips_data['cos_sim'] * 100
    
    # Находим существующие кадры изображений в директории
    existing_frames = []
    for i in range(100):  # Достаточно большой диапазон для поиска
        if list(seq_dir.glob(f"{sequence}_{i:03d}.png")):
            existing_frames.append(i)
    
    if not existing_frames:
        print(f"Error: No sequence frames found in {seq_dir}")
        return
    
    print(f"Found {len(existing_frames)} existing frames in {seq_dir}")
    
    # Определяем диапазон кадров
    if end_frame is None:
        end_frame = max(existing_frames)
    
    # Выбираем подмножество кадров для GIF с учетом step
    selected_frames = [f for f in existing_frames if start_frame <= f <= end_frame and (f - start_frame) % step == 0]
    
    print(f"Selected {len(selected_frames)} frames for animation")
    
    # Если включен режим smooth, создаем больше кадров, повторяя каждый кадр slow_factor раз
    if smooth:
        print(f"Using slow_factor {slow_factor} to create smoother animation")
        repeated_frames = []
        for frame in selected_frames:
            repeated_frames.extend([frame] * slow_factor)
        selected_frames = repeated_frames
    
    # Найдем минимальное и максимальное значение confidence среди всех моделей
    all_confidence_values = np.concatenate([
        baseline_data['confidence'].values,
        blurpool_data['confidence'].values,
        tips_data['confidence'].values
    ])
    min_confidence = np.min(all_confidence_values)
    max_confidence = np.max(all_confidence_values)
    
    print(f"Минимальное значение confidence: {min_confidence:.2f}%")
    print(f"Максимальное значение confidence: {max_confidence:.2f}%")
    
    # Получаем статистики для боксплотов
    model_stats = {
        'baseline': {
            'median': np.median(baseline_data['confidence']),
            'q1': np.percentile(baseline_data['confidence'], 25),
            'q3': np.percentile(baseline_data['confidence'], 75),
            'whislo': np.min(baseline_data['confidence']),
            'whishi': np.max(baseline_data['confidence'])
        },
        'blurpool': {
            'median': np.median(blurpool_data['confidence']),
            'q1': np.percentile(blurpool_data['confidence'], 25),
            'q3': np.percentile(blurpool_data['confidence'], 75),
            'whislo': np.min(blurpool_data['confidence']),
            'whishi': np.max(blurpool_data['confidence'])
        },
        'tips': {
            'median': np.median(tips_data['confidence']),
            'q1': np.percentile(tips_data['confidence'], 25),
            'q3': np.percentile(tips_data['confidence'], 75),
            'whislo': np.min(tips_data['confidence']),
            'whishi': np.max(tips_data['confidence'])
        }
    }
    
    # Очистка и создание временного каталога для изображений
    tmp_dir = Path(out_path).parent / "tmp_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"Генерация {len(selected_frames)} кадров анимации...")
    
    # Создаем отдельный кадр для каждого шага
    for idx, frame_idx in enumerate(selected_frames):
        frame_files = list(seq_dir.glob(f"{sequence}_{frame_idx:03d}.png"))
        if not frame_files:
            print(f"Warning: Frame file not found for {sequence} frame {frame_idx}")
            continue
        
        # Загружаем изображение
        img_path = frame_files[0]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получаем текущие значения confidence для каждой модели
        baseline_row = baseline_data[baseline_data['frame'] == frame_idx]
        blurpool_row = blurpool_data[blurpool_data['frame'] == frame_idx]
        tips_row = tips_data[tips_data['frame'] == frame_idx]
        
        if len(baseline_row) == 0 or len(blurpool_row) == 0 or len(tips_row) == 0:
            print(f"Warning: Missing data for frame {frame_idx}, skipping")
            continue
        
        baseline_conf = baseline_row['confidence'].values[0]
        blurpool_conf = blurpool_row['confidence'].values[0]
        tips_conf = tips_row['confidence'].values[0]
        
        # Получаем цвета для текущих значений
        baseline_color = get_color(baseline_conf, min_confidence, max_confidence)
        blurpool_color = get_color(blurpool_conf, min_confidence, max_confidence)
        tips_color = get_color(tips_conf, min_confidence, max_confidence)
        
        # Создаем фигуру с двумя подграфиками (изображение и боксплот)
        fig, (ax_img, ax_box) = plt.subplots(1, 2, figsize=(16, 8), 
                                          gridspec_kw={'width_ratios': [1, 1]})
        
        # Отображаем изображение
        ax_img.imshow(img)
        ax_img.axis('off')
        
        # Настраиваем график боксплотов
        ax_box.set_ylim(y_min, 100)
        ax_box.set_xlim(0.5, 3.5)
        
        # Позиции боксплотов и их метки
        positions = [1, 2, 3]
        labels = ['Baseline', 'Anti-aliased', 'TIPS']
        models = ['baseline', 'blurpool', 'tips']
        current_values = [baseline_conf, blurpool_conf, tips_conf]
        colors = [baseline_color, blurpool_color, tips_color]
        
        # Добавляем цветные бины и боксплоты
        for i, (model, pos, value, color) in enumerate(zip(models, positions, current_values, colors)):
            stats = model_stats[model]
            
            # 1. Сначала рисуем цветной бин от y_min до текущего значения
            bin_height = value - y_min
            bin_rect = Rectangle(
                (pos - 0.4, y_min), 
                0.8, 
                bin_height,
                facecolor=color, 
                edgecolor=None,
                alpha=0.8  # Увеличиваем непрозрачность для более насыщенного цвета
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
                alpha=0.3  # Уменьшаем непрозрачность для большей прозрачности
            )
            ax_box.add_patch(box)
            
            # Медиана - делаем ее более заметной, но не такой непрозрачной
            ax_box.hlines(stats['median'], pos - 0.4, pos + 0.4, colors='black', linewidth=2.0, alpha=0.5)
            
            # Усы - делаем их более прозрачными
            ax_box.vlines(pos, stats['q1'], stats['whislo'], colors='black', linewidth=1.5, alpha=0.4)
            ax_box.vlines(pos, stats['q3'], stats['whishi'], colors='black', linewidth=1.5, alpha=0.4)
            ax_box.hlines(stats['whislo'], pos - 0.2, pos + 0.2, colors='black', linewidth=1.5, alpha=0.4)
            ax_box.hlines(stats['whishi'], pos - 0.2, pos + 0.2, colors='black', linewidth=1.5, alpha=0.4)
            
            # Подпись с текущим значением
            ax_box.text(pos, value + 1.5, f"{value:.1f}%", 
                      ha='center', va='bottom', fontsize=12, fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Настройка осей и подписей
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(labels, fontsize=12)
        ax_box.set_ylabel('Confidence (%)', fontsize=14)
        ax_box.set_title(f'Confidence for: sparrow', fontsize=16)
        ax_box.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Добавляем информацию о кадре
        # Если включен режим smooth, показываем процент пути
        total_frames = len(existing_frames)
        current_frame_number = existing_frames.index(frame_idx) if frame_idx in existing_frames else 0
        progress = current_frame_number / max(1, (total_frames - 1))
        frame_label = f"Frame: {frame_idx} (Progress: {progress*100:.1f}%)"
        fig.suptitle(frame_label, fontsize=14, y=0.95)
        
        plt.tight_layout()
        
        # Сохраняем изображение
        tmp_frame_path = tmp_dir / f"frame_{idx:04d}.png"
        plt.savefig(tmp_frame_path, bbox_inches='tight', pad_inches=0.1)
        plt.close('all')
    
    # Создаем GIF, используя одинаковый размер для всех кадров
    try:
        frames_for_gif = []
        frame_paths = sorted(list(tmp_dir.glob("frame_*.png")))
        
        # Загружаем первый кадр, чтобы определить его размер
        if frame_paths:
            first_frame = cv2.imread(str(frame_paths[0]))
            target_shape = first_frame.shape[:2]
            
            # Загружаем и изменяем размер всех кадров
            for frame_path in frame_paths:
                # Загружаем кадр
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Изменяем размер, если необходимо
                if frame.shape[:2] != target_shape:
                    frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
                
                frames_for_gif.append(frame)
            
            # Сохраняем GIF
            if frames_for_gif:
                imageio.mimsave(out_path, frames_for_gif, fps=fps, loop=loop)
                print(f"Boxplot GIF animation saved to {out_path}")
            else:
                print("Error: No frames were processed for the GIF")
        else:
            print("Error: No frame files found")
            
    except Exception as e:
        print(f"Error creating GIF: {str(e)}")
    
    # Удаляем временную директорию
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory: {e}")
        
def main():
    args = parse_args()
    create_boxplot_confidence_gif(
        args.seq_dir, 
        args.results_dir, 
        args.sequence, 
        args.model_base, 
        args.out_path, 
        args.fps, 
        args.start_frame, 
        args.end_frame, 
        args.step, 
        args.loop,
        args.y_min,
        args.smooth,
        args.slow_factor
    )

if __name__ == "__main__":
    main() 