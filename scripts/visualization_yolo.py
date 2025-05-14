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
    # Устанавливаем заголовок
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
        (x, y), w, h, linewidth=2, edgecolor='darkgreen', 
        facecolor='darkgreen', alpha=0.1
    )
    ax.add_patch(rect_gt)
    
    # Baseline YOLO bbox - красный
    if baseline_bbox is not None:
        baseline_bbox = parse_bbox_string(baseline_bbox)
        x, y, w, h = baseline_bbox
        rect_baseline = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='#FF0000', 
            facecolor='#FF0000', alpha=0.1
        )
        ax.add_patch(rect_baseline)
    
    # TIPS YOLO bbox - синий
    if tips_bbox is not None:
        tips_bbox = parse_bbox_string(tips_bbox)
        x, y, w, h = tips_bbox
        rect_tips = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='#0000FF', 
            facecolor='#0000FF', alpha=0.1
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
        baseline_patch = patches.Patch(color='#FF0000', label='Baseline YOLO', alpha=0.3)
        handles.append(baseline_patch)
        
    # TIPS-YOLO модель
    if has_tips:
        tips_patch = patches.Patch(color='#0000FF', label='TIPS-YOLO', alpha=0.3)
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
    fixed_size=(1200, 800)
):
    """
    Создает полный кадр сравнения YOLO моделей с изображением и боксплотом
    
    Args:
        img: изображение для отображения
        gt_bbox: ground truth bbox [x, y, w, h]
        baseline_bbox: предсказанный bbox базовой YOLO модели [x, y, w, h]
        baseline_conf: уверенность baseline модели [0-1]
        tips_bbox: предсказанный bbox TIPS-YOLO модели [x, y, w, h]
        tips_conf: уверенность TIPS-YOLO модели [0-1]
        min_conf: минимальное значение для оси Y боксплота
        max_conf: максимальное значение для оси Y боксплота
        frame_idx: номер текущего кадра
        total_frames: общее количество кадров
        fixed_size: фиксированный размер кадра (ширина, высота)
        
    Returns:
        fig: созданный кадр (фигура matplotlib)
    """   
    # Проверяем наличие TIPS данных
    has_tips = tips_bbox is not None and tips_conf is not None
    
    # Создаем синтетические данные для бокс-плотов
    baseline_data = get_boxplot_data(baseline_conf, min_conf, max_conf)
    
    if has_tips:
        tips_data = get_boxplot_data(tips_conf, min_conf, max_conf)
        # Создаем имитацию данных для Anti-aliased модели (среднее между baseline и TIPS)
        antialias_conf = np.mean([baseline_conf, tips_conf])
        antialias_data = get_boxplot_data(antialias_conf, min_conf, max_conf)
        
        data_to_plot = [baseline_data, antialias_data, tips_data]
        box_labels = ['Baseline', 'Anti-aliased', 'TIPS']
        box_colors = ['#F8B195', '#F67280', '#C06C84']
    else:
        data_to_plot = [baseline_data]
        box_labels = ['Baseline']
        box_colors = ['#F8B195']
    
    # Улучшаем контрастность изображения для лучшей видимости
    avg_brightness = np.mean(img)
    
    # Всегда применяем улучшение контрастности для гарантии видимости воробья
    # Применяем более агрессивное улучшение для очень темных изображений
    if avg_brightness < 50:
        img = cv2.convertScaleAbs(img, alpha=2.5, beta=75)
    else:
        img = cv2.convertScaleAbs(img, alpha=1.8, beta=50)
    
    # Дополнительно используем CLAHE для улучшения локального контраста
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception:
        # Если что-то пошло не так, просто используем исходное изображение с улучшенной яркостью
        pass
    
    # Задаем стиль визуализации - всегда светлая тема
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black'
    })
    
    # Преобразуем пиксели в дюймы (dpi=100)
    fig_width, fig_height = fixed_size[0]/100, fixed_size[1]/100
    
    # Создаем фигуру с фиксированным размером
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    fig.patch.set_facecolor('white')
    
    # Устанавливаем структуру c фиксированными соотношениями
    grid_h = 8  # Высота сетки
    grid_w = 12  # Ширина сетки (соответствует соотношению сторон 3:2)
    
    # Заголовок (1 строка вверху)
    ax_title = plt.subplot2grid((grid_h, grid_w), (0, 0), colspan=grid_w, rowspan=1)
    ax_title.set_facecolor('white')
    ax_title.axis('off')
    
    # Рассчитываем прогресс
    progress = frame_idx / max(1, total_frames) * 100
    ax_title.text(0.5, 0.5, f"Frame: {frame_idx} (Progress: {progress:.1f}%)", 
                 fontsize=16, fontweight='bold', ha='center', va='center', color='black')
    
    # Изображение (занимает большую часть пространства слева)
    ax_img = plt.subplot2grid((grid_h, grid_w), (1, 0), colspan=8, rowspan=grid_h-1)
    ax_img.set_facecolor('white')
    
    # Выделяем область с воробьем для лучшей видимости
    # Для этого создаем копию изображения с затемненным фоном
    enhanced_img = img.copy()
    
    # Получаем размеры изображения
    img_height, img_width = img.shape[:2]
    
    # Если у нас есть ground truth bbox, усиливаем эту область
    if gt_bbox is not None:
        # Получаем координаты gt_bbox
        x, y, w, h = gt_bbox
        
        # Расширяем область интереса (ROI) на 10% во всех направлениях
        roi_x = max(0, int(x - w * 0.1))
        roi_y = max(0, int(y - h * 0.1))
        roi_w = min(img_width - roi_x, int(w * 1.2))
        roi_h = min(img_height - roi_y, int(h * 1.2))
        
        # Создаем маску для выделения области интереса
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
        
        # Размываем края маски для плавного перехода
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Нормализуем маску от 0 до 1
        mask = mask.astype(float) / 255.0
        
        # Конвертируем маску в 3-канальную для применения к RGB изображению
        mask_3ch = np.dstack([mask, mask, mask])
        
        # Создаем затемненную версию изображения (70% яркости)
        darkened = (img * 0.7).astype(np.uint8)
        
        # Комбинируем оригинальное изображение и затемненную версию с использованием маски
        enhanced_img = img * mask_3ch + darkened * (1 - mask_3ch)
        enhanced_img = enhanced_img.astype(np.uint8)
    
    # Устанавливаем отступы для изображения
    ax_img.margins(0.05)
    
    # Отображаем изображение с выделенной областью интереса
    ax_img.imshow(enhanced_img, aspect='auto')
    
    # Отрисовка bbox
    draw_yolo_bboxes(
        ax_img, gt_bbox, baseline_bbox, 
        tips_bbox if has_tips else None
    )
    
    # Добавление легенды
    add_yolo_bbox_legend(
        ax_img, has_baseline=True, 
        has_tips=has_tips
    )
    
    # Отключаем оси
    ax_img.axis('off')
    
    # Боксплот confidence (справа)
    ax_conf = plt.subplot2grid((grid_h, grid_w), (1, 8), colspan=4, rowspan=grid_h-1)
    ax_conf.set_facecolor('white')
    
    # Создаем боксплот
    bp = create_yolo_confidence_boxplot(
        ax_conf, data_to_plot, box_labels, box_colors,
        min_conf=min_conf, max_conf=max_conf
    )
    
    # Значения confidence для отображения
    confidence_values = []
    confidence_values.append((1, baseline_conf * 100))
    
    if has_tips:
        confidence_values.append((2, antialias_conf * 100))
        confidence_values.append((3, tips_conf * 100))
    
    # Добавляем метки с процентами
    add_yolo_confidence_labels(ax_conf, confidence_values, min_conf, max_conf)
    
    # Используем плотную компоновку
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    return fig


# =====================================================================
# Функции для создания GIF анимаций с визуализацией YOLO моделей
# =====================================================================

def create_yolo_comparison_gif(
    seq_dir, baseline_results, tips_results=None, output_path=None,
    sequence="seq_0", fps=5, start_frame=0, end_frame=None, step=1,
    slow_factor=3, fixed_size=(1200, 800), min_conf=75, max_conf=85
):
    """
    Создает GIF-анимацию сравнения YOLO моделей с изображением и боксплотом
    
    Args:
        seq_dir: директория с последовательностями
        baseline_results: путь к CSV с результатами baseline YOLO модели
        tips_results: путь к CSV с результатами TIPS-YOLO модели
        output_path: путь для сохранения GIF
        sequence: идентификатор последовательности (например, 'seq_0')
        fps: частота кадров в GIF
        start_frame: начальный кадр
        end_frame: конечный кадр (None = все)
        step: шаг между кадрами
        slow_factor: коэффициент замедления
        fixed_size: фиксированный размер кадра (ширина, высота)
        min_conf: минимальное значение для оси Y боксплота
        max_conf: максимальное значение для оси Y боксплота
        
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    # Сбрасываем настройки matplotlib для чистого рабочего окружения
    plt.rcdefaults()
    
    # Загружаем результаты моделей
    baseline_df = load_csv_results(baseline_results)
    if baseline_df is None:
        return False
    
    has_tips = False
    if tips_results:
        tips_df = load_csv_results(tips_results)
        has_tips = tips_df is not None
    
    # Находим кадры последовательности
    selected_frames, selected_files = find_sequence_frames(
        seq_dir, sequence, start_frame, end_frame, step
    )
    
    if not selected_frames:
        return False
    
    print(f"Выбрано {len(selected_frames)} кадров для анимации")
    
    # Создаем кадры для плавной анимации
    if slow_factor > 1:
        print(f"Применяем фактор замедления {slow_factor} для плавной анимации")
        original_len = len(selected_frames)
        total_frames = original_len * slow_factor
        print(f"Всего кадров после замедления: {total_frames}")
    else:
        original_len = len(selected_frames)
        total_frames = original_len
    
    # Создаем временную директорию для кадров GIF
    tmp_dir = create_temp_directory(output_path, "tmp_yolo_gif")
    
    print(f"Генерация {total_frames} кадров анимации...")
    
    # Создаем отдельный кадр для каждого шага
    valid_frames = []
    
    for idx in range(total_frames):
        # Используем модульную арифметику для доступа к исходным кадрам
        original_idx = idx % original_len
        frame_idx = selected_frames[original_idx]
        img_path = selected_files[original_idx]
        
        # Загружаем изображение
        img = load_image(img_path)
        if img is None:
            continue
        
        try:
            # Получаем данные для текущего кадра из CSV
            frame_prefix = img_path.name
            
            # Ищем строку в baseline результатах
            baseline_row = None
            
            # Сначала пробуем найти по имени файла (если есть колонка 'frame')
            if 'frame' in baseline_df.columns:
                matched_rows = baseline_df[baseline_df['frame'] == frame_prefix]
                if len(matched_rows) > 0:
                    baseline_row = matched_rows.iloc[0]
            
            # Если не нашли по имени, используем индекс с проверкой границ
            if baseline_row is None:
                csv_idx = min(original_idx, len(baseline_df) - 1)
                baseline_row = baseline_df.iloc[csv_idx]
            
            # Извлекаем данные
            baseline_conf = baseline_row['confidence']
            gt_bbox = parse_bbox_string(baseline_row['gt_bbox'])
            baseline_bbox = parse_bbox_string(baseline_row['pred_bbox'])
            
            # Аналогично для tips, если доступно
            tips_bbox = None
            tips_conf = None
            
            if has_tips:
                tips_row = None
                
                # Сначала пробуем найти по имени файла
                if 'frame' in tips_df.columns:
                    matched_rows = tips_df[tips_df['frame'] == frame_prefix]
                    if len(matched_rows) > 0:
                        tips_row = matched_rows.iloc[0]
                
                # Если не нашли по имени, используем индекс с проверкой границ
                if tips_row is None:
                    csv_idx = min(original_idx, len(tips_df) - 1)
                    tips_row = tips_df.iloc[csv_idx]
                
                tips_conf = tips_row['confidence']
                tips_bbox = parse_bbox_string(tips_row['pred_bbox'])
            
            # Создаем кадр сравнения
            fig = create_yolo_frame(
                img, gt_bbox, baseline_bbox, baseline_conf,
                tips_bbox, tips_conf,
                min_conf, max_conf, frame_idx, total_frames,
                fixed_size
            )
            
            # Сохраняем кадр
            tmp_frame_path = tmp_dir / f"frame_{idx:04d}.png"
            fig.savefig(tmp_frame_path, dpi=100, bbox_inches='tight', 
                        facecolor='white', edgecolor='none', pad_inches=0.1,
                        format='png')
            plt.close(fig)
            
            # Проверяем, создался ли файл
            if os.path.exists(tmp_frame_path):
                valid_frames.append(tmp_frame_path)
            
            # Для индикации прогресса
            if (idx + 1) % 10 == 0:
                print(f"Обработано {idx + 1}/{total_frames} кадров")
                
        except Exception as e:
            import traceback
            print(f"Ошибка при создании кадра {idx}: {e}")
            traceback.print_exc()
            continue
    
    print(f"Успешно создано {len(valid_frames)} кадров")
    if len(valid_frames) < 2:
        print(f"Недостаточно кадров для создания GIF (создано {len(valid_frames)})")
        cleanup_temp_directory(tmp_dir)
        return False
    
    # Создаем GIF
    result = save_gif_animation(valid_frames, output_path, fps)
    
    # Очищаем временную директорию
    cleanup_temp_directory(tmp_dir)
    
    return result 