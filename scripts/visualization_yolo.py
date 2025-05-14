#!/usr/bin/env python3
"""
Модуль для создания визуализаций YOLO моделей.
Содержит функции для визуализации детекций YOLO и создания GIF-анимаций.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from pathlib import Path
import cv2
from PIL import Image

# Импортируем общие утилиты
from scripts.utils import (
    load_image, find_sequence_frames, create_frames_for_gif, save_gif_animation,
    create_temp_directory, cleanup_temp_directory, get_boxplot_data, get_color_by_value,
    parse_bbox_string, load_csv_results
)


# =====================================================================
# Визуализация для боксплотов уверенности YOLO
# =====================================================================

def create_yolo_confidence_boxplot(
    ax, data_to_plot, labels, colors, min_conf=75, max_conf=85,
    title="Confidence for: sparrow"
):
    """
    Создает бокс-плот уверенности YOLO модели(ей)
    
    Args:
        ax: оси matplotlib для рисования
        data_to_plot: данные для боксплота (список списков)
        labels: метки для каждого бокса
        colors: цвета для каждого бокса
        min_conf: минимальное значение для оси Y
        max_conf: максимальное значение для оси Y
        title: заголовок графика
        
    Returns:
        bp: боксплот (для дальнейшей настройки)
    """
    # Устанавливаем заголовок только если он не пустой
    if title:
        ax.set_title(title, fontsize=14, color="black")
    
    # Устанавливаем границы для оси Y
    ax.set_ylim(min_conf, max_conf)
    ax.set_ylabel("Confidence (%)", fontsize=12, color="black")
    
    # Создаем бокс-плот с настроенными параметрами
    bp = ax.boxplot(
        data_to_plot, notch=False, patch_artist=True, 
        vert=True, widths=0.4, showfliers=False, 
        medianprops={'color': 'white', 'linewidth': 2},
        boxprops={'linewidth': 1.5},
        whiskerprops={'linewidth': 1.5, 'color': 'black'},
        capprops={'linewidth': 1.5, 'color': 'black'}
    )
    
    # Устанавливаем метки оси X
    ax.set_xticklabels(labels, color="black")
    
    # Раскрашиваем боксы в соответствующие цвета
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Настраиваем внешний вид графика
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    return bp


def add_yolo_confidence_labels(ax, confidence_values, min_conf=75, max_conf=85):
    """
    Добавляет значения уверенности к боксплоту YOLO
    
    Args:
        ax: оси matplotlib
        confidence_values: список кортежей (позиция_x, значение)
        min_conf: минимальное значение для оси Y
        max_conf: максимальное значение для оси Y
    """    
    for x_pos, value in confidence_values:
        # Гарантируем фиксированное расстояние над боксом
        text_y = value + 2  # Фиксированное смещение в 2 процентных пункта
        va = 'bottom'
        
        # Проверяем, не выходит ли значение за пределы графика
        if text_y > max_conf - 1:
            text_y = value - 2  # Если выходит за верхний предел, помещаем ниже
            va = 'top'
        
        # Добавляем текст с контрастным фоном для лучшей видимости
        ax.text(
            x_pos, text_y, f"{value:.1f}%", 
            ha='center', va=va, color="black", fontweight='bold',
            bbox=dict(facecolor="white", alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )


# =====================================================================
# Функции для отрисовки bbox детекций YOLO
# =====================================================================

def draw_yolo_bboxes(
    ax, gt_bbox, baseline_bbox=None, tips_bbox=None
):
    """
    Отрисовывает bbox детекций YOLO на изображении
    
    Args:
        ax: оси matplotlib для рисования
        gt_bbox: ground truth bbox [x, y, w, h]
        baseline_bbox: предсказанный bbox базовой YOLO модели [x, y, w, h]
        tips_bbox: предсказанный bbox TIPS-YOLO модели [x, y, w, h]
    """
    # Парсим строковые представления bbox, если необходимо
    gt_bbox = parse_bbox_string(gt_bbox)
    
    # Ground truth bbox - зеленый
    x, y, w, h = gt_bbox
    rect_gt = patches.Rectangle(
        (x, y), w, h, linewidth=3, edgecolor='darkgreen', 
        facecolor='none', alpha=0.8
    )
    ax.add_patch(rect_gt)
    
    # Добавляем дополнительную полупрозрачную заливку для выделения области объекта
    rect_gt_fill = patches.Rectangle(
        (x, y), w, h, linewidth=0, edgecolor='none', 
        facecolor='lightgreen', alpha=0.2
    )
    ax.add_patch(rect_gt_fill)
    
    # Baseline YOLO bbox - красный
    if baseline_bbox is not None:
        baseline_bbox = parse_bbox_string(baseline_bbox)
        x, y, w, h = baseline_bbox
        rect_baseline = patches.Rectangle(
            (x, y), w, h, linewidth=3, edgecolor='red', 
            facecolor='none', alpha=0.8
        )
        ax.add_patch(rect_baseline)
    
    # TIPS YOLO bbox - синий
    if tips_bbox is not None:
        tips_bbox = parse_bbox_string(tips_bbox)
        x, y, w, h = tips_bbox
        rect_tips = patches.Rectangle(
            (x, y), w, h, linewidth=3, edgecolor='blue', 
            facecolor='none', alpha=0.8
        )
        ax.add_patch(rect_tips)


def add_yolo_bbox_legend(
    ax, has_baseline=True, has_tips=False
):
    """
    Добавляет легенду для боксов детекций YOLO
    
    Args:
        ax: оси matplotlib для рисования
        has_baseline: наличие baseline YOLO bbox
        has_tips: наличие TIPS YOLO bbox
    """
    handles = []
    
    # Ground Truth всегда присутствует
    gt_patch = patches.Patch(color='darkgreen', label='Ground Truth', alpha=0.3)
    handles.append(gt_patch)
        
    # Базовая YOLO модель
    if has_baseline:
        baseline_patch = patches.Patch(color='red', label='Baseline YOLO', alpha=0.3)
        handles.append(baseline_patch)
        
    # TIPS-YOLO модель
    if has_tips:
        tips_patch = patches.Patch(color='blue', label='TIPS-YOLO', alpha=0.3)
        handles.append(tips_patch)
    
    if handles:
        ax.legend(handles=handles, loc='lower right', framealpha=0.8)


# =====================================================================
# Функции для создания полных визуализаций кадров YOLO
# =====================================================================

def create_yolo_frame(
    img, gt_bbox, baseline_bbox, baseline_conf, 
    tips_bbox=None, tips_conf=None, 
    min_conf=75, max_conf=85, frame_idx=0, total_frames=100,
    all_frames=None, fixed_size=(16, 8), object_class="object",
    img_filename=""
):
    """
    Создает полный кадр сравнения YOLO моделей с изображением и боксплотом в стиле legacy
    
    Args:
        img: изображение для отображения
        gt_bbox: ground truth bbox [x, y, w, h]
        baseline_bbox: предсказанный bbox базовой YOLO модели [x, y, w, h]
        baseline_conf: уверенность baseline модели [0-100]
        tips_bbox: предсказанный bbox TIPS-YOLO модели [x, y, w, h]
        tips_conf: уверенность TIPS-YOLO модели [0-100]
        min_conf: минимальное значение для оси Y боксплота
        max_conf: максимальное значение для оси Y боксплота
        frame_idx: номер текущего кадра
        total_frames: общее количество кадров
        all_frames: список всех номеров кадров
        fixed_size: фиксированный размер фигуры (ширина, высота) в дюймах
        object_class: класс объекта
        img_filename: имя файла изображения для отображения в заголовке
        
    Returns:
        fig: созданный кадр (фигура matplotlib)
    """
    if all_frames is None:
        all_frames = list(range(total_frames))
    
    # Проверяем наличие TIPS данных
    has_tips = tips_bbox is not None and tips_conf is not None
    
    # Создаем фигуру с двумя подграфиками (изображение и боксплот)
    fig, (ax_img, ax_box) = plt.subplots(1, 2, figsize=fixed_size, 
                                      gridspec_kw={'width_ratios': [1, 1]})
    
    # Отображаем изображение
    ax_img.imshow(img)
    ax_img.axis('off')
    
    # Отрисовываем bounding box на изображении
    draw_yolo_bboxes(ax_img, gt_bbox, baseline_bbox, tips_bbox if has_tips else None)
    add_yolo_bbox_legend(ax_img, True, has_tips)
    
    # Подготовка данных для боксплотов
    # Определяем модели и данные
    if has_tips:
        model_names = ['Baseline', 'TIPS']
        models_data = {
            'Baseline': get_boxplot_data(baseline_conf, min_conf, max_conf),
            'TIPS': get_boxplot_data(tips_conf, min_conf, max_conf)
        }
        
        models_current = {
            'Baseline': baseline_conf,
            'TIPS': tips_conf
        }
    else:
        model_names = ['Baseline']
        models_data = {
            'Baseline': get_boxplot_data(baseline_conf, min_conf, max_conf)
        }
        models_current = {
            'Baseline': baseline_conf
        }
    
    # Настраиваем график боксплотов
    ax_box.set_ylim(min_conf, max_conf)
    ax_box.set_xlim(0.5, len(model_names) + 0.5)
    
    # Позиции боксплотов
    positions = list(range(1, len(model_names) + 1))
    
    # Получаем статистики для боксплотов
    model_stats = {}
    for model in model_names:
        values = models_data[model]
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
        value = models_current[model]
        
        # Обрабатываем случай, когда значение ниже минимального порога на графике
        if value < min_conf:
            # Отображаем специальный красный бин фиксированной высоты
            error_bin_height = 1.0  # 1% высоты для индикации низкого значения
            bin_rect = patches.Rectangle(
                (pos - 0.4, min_conf), 
                0.8, 
                error_bin_height,
                facecolor=(0.8, 0.0, 0.0),  # Темно-красный цвет
                edgecolor='black',
                alpha=0.8,
                linewidth=1.0,
                hatch='///'  # Штриховка для выделения
            )
            ax_box.add_patch(bin_rect)
            
            # Добавляем текстовую метку с реальным значением
            value_label = f"Low: {value:.1f}%"
            ax_box.text(pos, min_conf - 2, value_label, 
                      ha='center', va='top', fontsize=10, fontweight='bold', color='red',
                      bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2'))
            
            # Добавляем вертикальную пунктирную линию для подсветки модели с низким confidence
            ax_box.axvline(x=pos, ymin=0, ymax=1, color='red', alpha=0.3, linestyle='--')
        else:
            # Получаем цвет бина в зависимости от значения
            color = get_color_by_value(value, min_conf, max_conf, "confidence")
            
            # 1. Сначала рисуем цветной бин от min_value до текущего значения
            bin_height = value - min_conf
            bin_rect = patches.Rectangle(
                (pos - 0.4, min_conf), 
                0.8, 
                bin_height,
                facecolor=color, 
                edgecolor=None,
                alpha=0.8  # Непрозрачность как в legacy коде
            )
            ax_box.add_patch(bin_rect)
            
            # Подпись с текущим значением
            value_label = f"{value:.1f}%"
            ax_box.text(pos, value + (max_conf - min_conf) * 0.02, value_label, 
                      ha='center', va='bottom', fontsize=12, fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
        # 2. Затем рисуем стандартный боксплот поверх
        # Бокс
        box_height = stats['q3'] - stats['q1']
        box = patches.Rectangle(
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
    
    # Настройка осей и подписей
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(model_names, fontsize=12)
    ax_box.set_ylabel("Confidence (%)", fontsize=14)
    # Добавляем имя файла в заголовок для проверки синхронизации
    title_text = f"Confidence for: {object_class}"
    if img_filename:
        title_text += f" | {img_filename}"
    ax_box.set_title(title_text, fontsize=16)
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
    
    # Вместо plt.tight_layout() используем ручную настройку расположения осей
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95, wspace=0.2, hspace=0.2)
    
    return fig


# =====================================================================
# Функции для создания GIF анимаций с визуализацией YOLO моделей
# =====================================================================

def create_yolo_comparison_gif(
    seq_dir, baseline_results, tips_results=None, output_path=None,
    sequence="seq_0", fps=5, start_frame=0, end_frame=None, step=1,
    slow_factor=3, fixed_size=(16, 8), min_conf=75, max_conf=85,
    object_class="object"
):
    """
    Создает GIF-анимацию сравнения YOLO моделей
    
    Args:
        seq_dir: директория с последовательностями изображений
        baseline_results: путь к CSV файлу с результатами базовой модели
        tips_results: путь к CSV файлу с результатами TIPS модели
        output_path: путь для сохранения GIF-анимации
        sequence: ID последовательности
        fps: кадров в секунду
        start_frame: начальный кадр
        end_frame: конечный кадр
        step: шаг между кадрами
        slow_factor: коэффициент замедления (повторяет кадры)
        fixed_size: фиксированный размер кадра (ширина, высота) в дюймах
        min_conf: минимальное значение для оси Y боксплота (%)
        max_conf: максимальное значение для оси Y боксплота (%)
        object_class: класс объекта для отображения в заголовке
        
    Returns:
        str: путь к созданной GIF-анимации
    """
    # Преобразуем пути
    seq_dir = Path(seq_dir)
    
    # Формируем имя выходного файла, если не указано
    if output_path is None:
        model_part = "baseline"
        if tips_results:
            model_part += "_vs_tips"
        
        output_path = f"yolo_{model_part}_{sequence}.gif"
    
    # Загружаем результаты
    print(f"Загрузка результатов модели baseline из {baseline_results}")
    baseline_data = load_csv_results(baseline_results)
    
    # Проверяем наличие TIPS результатов
    if tips_results:
        print(f"Загрузка результатов модели TIPS из {tips_results}")
        tips_data = load_csv_results(tips_results)
    else:
        tips_data = None
    
    # Находим все кадры в последовательности
    selected_frames, selected_files = find_sequence_frames(seq_dir, sequence, start_frame, end_frame, step)
    print(f"Найдено {len(selected_frames)} кадров в последовательности {sequence}")
    
    # Ничего не найдено - завершаем работу
    if not selected_frames:
        print(f"Ошибка: Не найдены изображения последовательности {sequence} в {seq_dir}")
        return None
    
    print(f"Выбрано {len(selected_frames)} кадров для анимации")
    
    if not selected_frames:
        print("Ошибка: Не найдены кадры для выбранного диапазона")
        return None
    
    # Создаем временную директорию для кадров
    temp_dir = create_temp_directory("yolo_gif")
    
    # Создаем и сохраняем кадры
    print("Создание кадров анимации...")
    frames_for_gif = []
    
    for i, frame_idx in enumerate(selected_frames):
        print(f"Обработка кадра {frame_idx} ({i+1}/{len(selected_frames)})", end="\r")
        
        # Загружаем изображение
        img_path = selected_files[i]
        img_filename = os.path.basename(img_path)
        if not os.path.exists(img_path):
            print(f"Пропуск: Не найден файл {img_path}")
            continue
        
        img = load_image(img_path)
        
        # Получаем данные для базовой модели, используя имя файла
        try:
            baseline_row = baseline_data[baseline_data['frame'] == img_filename].iloc[0]
            baseline_bbox = baseline_row['pred_bbox']  # Используем pred_bbox вместо bbox
            baseline_conf = baseline_row['confidence'] * 100  # Преобразуем в проценты
            gt_bbox = baseline_row['gt_bbox']
        except (IndexError, KeyError) as e:
            print(f"Ошибка: Не найдены данные для кадра {img_filename} в baseline: {e}")
            # Проверяем соответствие значений в 'frame'
            if 'frame' in baseline_data.columns:
                print(f"Доступные значения в столбце 'frame': {baseline_data['frame'].unique()[:5]}...")
            continue
        
        # Получаем данные для TIPS модели, если доступны
        if tips_data is not None:
            try:
                tips_row = tips_data[tips_data['frame'] == img_filename].iloc[0]
                tips_bbox = tips_row['pred_bbox']  # Используем pred_bbox вместо bbox
                tips_conf = tips_row['confidence'] * 100  # Преобразуем в проценты
            except (IndexError, KeyError) as e:
                print(f"Ошибка: Не найдены данные для кадра {img_filename} в TIPS: {e}")
                tips_bbox = None
                tips_conf = None
        else:
            tips_bbox = None
            tips_conf = None
        
        # Создаем кадр
        fig = create_yolo_frame(
            img, gt_bbox, baseline_bbox, baseline_conf,
            tips_bbox, tips_conf,
            min_conf, max_conf, frame_idx, len(selected_frames),
            all_frames=selected_frames, fixed_size=fixed_size,
            object_class=object_class, img_filename=img_filename
        )
        
        # Сохраняем кадр
        frame_path = Path(temp_dir) / f"frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        # Добавляем путь кадра в список
        frames_for_gif.append(frame_path)
    
    print("\nВсе кадры созданы!")
    
    # Создаем GIF
    if frames_for_gif:
        # Если нужно замедление, дублируем кадры
        if slow_factor > 1:
            print(f"Применение slow_factor={slow_factor} для замедления анимации")
            frames_for_gif = create_frames_for_gif(frames_for_gif, slow_factor=slow_factor)
        
        gif_path = save_gif_animation(frames_for_gif, output_path, fps=fps)
        print(f"GIF-анимация сохранена в {gif_path}")
    else:
        gif_path = None
        print("Ошибка: Не удалось создать кадры для GIF")
    
    # Очищаем временные файлы
    cleanup_temp_directory(temp_dir)
    
    return gif_path 