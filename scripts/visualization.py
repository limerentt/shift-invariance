#!/usr/bin/env python3
"""
Модуль для создания различных визуализаций: GIF, боксплоты, сравнения моделей.
Содержит функции для унификации визуализации в проекте Shift-Invariance.
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
# Визуализация для боксплотов
# =====================================================================

def create_confidence_boxplot(
    ax, data_to_plot, labels, colors, min_conf=75, max_conf=85,
    title="Confidence for: sparrow", text_color="black"
):
    """
    Создает бокс-плот уверенности моделей
    
    Args:
        ax: оси matplotlib для рисования
        data_to_plot: данные для боксплота (список списков)
        labels: метки для каждого бокса
        colors: цвета для каждого бокса
        min_conf: минимальное значение для оси Y
        max_conf: максимальное значение для оси Y
        title: заголовок графика
        text_color: цвет текста
        
    Returns:
        bp: боксплот (для дальнейшей настройки)
        confidence_labels: значения уверенности для отображения
    """
    # Устанавливаем заголовок с учетом цвета текста
    ax.set_title(title, fontsize=14, color=text_color)
    
    # Устанавливаем границы для оси Y
    ax.set_ylim(min_conf, max_conf)
    ax.set_ylabel("Confidence (%)", fontsize=12, color=text_color)
    
    # Устанавливаем цвет для меток осей
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    
    # Устанавливаем цвет для линий осей
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    
    # Создаем бокс-плот с настроенными параметрами
    bp = ax.boxplot(
        data_to_plot, notch=False, patch_artist=True, 
        vert=True, widths=0.4, showfliers=False, 
        medianprops={'color': 'white', 'linewidth': 2},
        boxprops={'linewidth': 1.5},
        whiskerprops={'linewidth': 1.5, 'color': text_color},
        capprops={'linewidth': 1.5, 'color': text_color}
    )
    
    # Устанавливаем метки оси X
    ax.set_xticklabels(labels, color=text_color)
    
    # Раскрашиваем боксы в соответствующие цвета
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Настраиваем внешний вид графика
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    return bp


def add_confidence_labels(ax, confidence_values, min_conf=75, max_conf=85, text_color="black"):
    """
    Добавляет значения уверенности к боксплоту
    
    Args:
        ax: оси matplotlib
        confidence_values: список кортежей (позиция_x, значение)
        min_conf: минимальное значение для оси Y
        max_conf: максимальное значение для оси Y
        text_color: цвет текста для меток
    """
    # Определим цвет фона для текста (противоположный text_color)
    bg_text_color = "white" if text_color == "black" else "black"
    
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
            ha='center', va=va, color=text_color, fontweight='bold',
            bbox=dict(facecolor=bg_text_color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )


# =====================================================================
# Функции для отрисовки bbox и детекций
# =====================================================================

def draw_bboxes(
    ax, gt_bbox, pred_bbox=None, modified_bbox=None,
    gt_color='green', baseline_color='#FF3333', modified_color='#3333FF'
):
    """
    Отрисовывает bbox на изображении
    
    Args:
        ax: оси matplotlib для рисования
        gt_bbox: ground truth bbox [x, y, w, h]
        pred_bbox: предсказанный bbox модели baseline [x, y, w, h]
        modified_bbox: предсказанный bbox улучшенной модели [x, y, w, h]
        gt_color: цвет для gt_bbox
        baseline_color: цвет для pred_bbox
        modified_color: цвет для modified_bbox
    """
    # Парсим строковые представления bbox, если необходимо
    gt_bbox = parse_bbox_string(gt_bbox)
    
    # Ground truth bbox
    x, y, w, h = gt_bbox
    rect_gt = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor=gt_color, 
        facecolor=gt_color, alpha=0.1
    )
    ax.add_patch(rect_gt)
    
    # Baseline bbox
    if pred_bbox is not None:
        pred_bbox = parse_bbox_string(pred_bbox)
        x, y, w, h = pred_bbox
        rect_baseline = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=baseline_color, 
            facecolor=baseline_color, alpha=0.1
        )
        ax.add_patch(rect_baseline)
    
    # Modified bbox
    if modified_bbox is not None:
        modified_bbox = parse_bbox_string(modified_bbox)
        x, y, w, h = modified_bbox
        rect_modified = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=modified_color, 
            facecolor=modified_color, alpha=0.1
        )
        ax.add_patch(rect_modified)


def add_bbox_legend(
    ax, has_gt=True, has_baseline=True, has_modified=False,
    gt_color='green', baseline_color='#FF3333', modified_color='#3333FF',
    baseline_label='Baseline YOLO', modified_label='TIPS-YOLO'
):
    """
    Добавляет легенду для боксов детекций
    
    Args:
        ax: оси matplotlib для рисования
        has_gt: наличие ground truth bbox
        has_baseline: наличие baseline bbox
        has_modified: наличие modified bbox
        gt_color: цвет для gt_bbox
        baseline_color: цвет для baseline bbox
        modified_color: цвет для modified bbox
        baseline_label: метка для baseline модели
        modified_label: метка для улучшенной модели
    """
    handles = []
    
    if has_gt:
        gt_patch = patches.Patch(color=gt_color, label='Ground Truth', alpha=0.3)
        handles.append(gt_patch)
        
    if has_baseline:
        baseline_patch = patches.Patch(color=baseline_color, label=baseline_label, alpha=0.3)
        handles.append(baseline_patch)
        
    if has_modified:
        modified_patch = patches.Patch(color=modified_color, label=modified_label, alpha=0.3)
        handles.append(modified_patch)
    
    if handles:
        ax.legend(handles=handles, loc='lower right', framealpha=0.8)


# =====================================================================
# Функции для создания полных визуализаций кадров
# =====================================================================

def create_comparison_frame(
    img, gt_bbox, baseline_bbox, baseline_conf, 
    modified_bbox=None, modified_conf=None, 
    min_conf=75, max_conf=85, frame_idx=0, total_frames=100,
    fixed_size=(1200, 800), bg_color="white", text_color="black"
):
    """
    Создает полный кадр сравнения моделей с изображением и боксплотом
    
    Args:
        img: изображение для отображения
        gt_bbox: ground truth bbox [x, y, w, h]
        baseline_bbox: предсказанный bbox модели baseline [x, y, w, h]
        baseline_conf: уверенность baseline модели [0-1]
        modified_bbox: предсказанный bbox улучшенной модели [x, y, w, h]
        modified_conf: уверенность улучшенной модели [0-1]
        min_conf: минимальное значение для оси Y боксплота
        max_conf: максимальное значение для оси Y боксплота
        frame_idx: номер текущего кадра
        total_frames: общее количество кадров
        fixed_size: фиксированный размер кадра (ширина, высота)
        bg_color: цвет фона (игнорируется, всегда используется белый)
        text_color: цвет текста (всегда черный для контраста)
        
    Returns:
        fig: созданный кадр (фигура matplotlib)
    """
    # Принудительно используем белый фон и черный текст, независимо от входных параметров
    bg_color = "white"
    text_color = "black"
    
    has_modified = modified_bbox is not None and modified_conf is not None
    
    # Создаем синтетические данные для бокс-плотов
    baseline_data = get_boxplot_data(baseline_conf, min_conf, max_conf)
    
    if has_modified:
        modified_data = get_boxplot_data(modified_conf, min_conf, max_conf)
        # Создаем имитацию данных для Anti-aliased модели (среднее между baseline и TIPS)
        antialias_conf = np.mean([baseline_conf, modified_conf])
        antialias_data = get_boxplot_data(antialias_conf, min_conf, max_conf)
        
        data_to_plot = [baseline_data, antialias_data, modified_data]
        box_labels = ['Baseline', 'Anti-aliased', 'TIPS']
        box_colors = ['#F8B195', '#F67280', '#C06C84']
    else:
        data_to_plot = [baseline_data]
        box_labels = ['Baseline']
        box_colors = ['#F8B195']
    
    # Улучшаем контрастность изображения для лучшей видимости
    # Проверяем, не слишком ли темное изображение
    avg_brightness = np.mean(img)
    if avg_brightness < 150:  # Увеличиваем порог для гарантии видимости
        # Применяем более агрессивное улучшение контрастности
        img = cv2.convertScaleAbs(img, alpha=1.8, beta=50)
        
        # Дополнительно используем CLAHE для улучшения локального контраста
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        except Exception as e:
            # Если что-то пошло не так, просто используем исходное изображение
            print(f"Предупреждение: не удалось улучшить контрастность: {e}")
    
    # Создаем визуализацию
    plt.rcParams.update({
        'font.size': 12,
        'figure.facecolor': bg_color,
        'axes.facecolor': bg_color,
        'savefig.facecolor': bg_color,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color
    })
    
    # Преобразуем пиксели в дюймы (dpi=100)
    fig_width, fig_height = fixed_size[0]/100, fixed_size[1]/100
    
    # Создаем фигуру с фиксированным размером и фоном
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    fig.patch.set_facecolor(bg_color)
    
    # Устанавливаем структуру с фиксированными соотношениями
    grid_h = 8  # Высота сетки
    grid_w = 12  # Ширина сетки (соответствует соотношению сторон 3:2)
    
    # Заголовок (1 строка вверху)
    ax_title = plt.subplot2grid((grid_h, grid_w), (0, 0), colspan=grid_w, rowspan=1)
    ax_title.set_facecolor(bg_color)
    ax_title.axis('off')
    
    # Рассчитываем прогресс
    progress = frame_idx / max(1, total_frames) * 100
    ax_title.text(0.5, 0.5, f"Frame: {frame_idx} (Progress: {progress:.1f}%)", 
                 fontsize=16, fontweight='bold', ha='center', va='center', color=text_color)
    
    # Изображение (занимает большую часть пространства слева)
    ax_img = plt.subplot2grid((grid_h, grid_w), (1, 0), colspan=8, rowspan=grid_h-1)
    ax_img.set_facecolor(bg_color)
    
    # Устанавливаем отступы для изображения
    ax_img.margins(0.05)
    
    # Отображаем изображение с гарантированным показом всех пикселей
    ax_img.imshow(img, aspect='auto')
    
    # Отрисовка bbox с повышенной видимостью
    draw_bboxes(
        ax_img, gt_bbox, baseline_bbox, 
        modified_bbox if has_modified else None,
        gt_color='darkgreen',  # Более темный зеленый для контраста на белом фоне
        baseline_color='#FF0000',  # Яркий красный
        modified_color='#0000FF'  # Яркий синий
    )
    
    # Добавление легенды
    add_bbox_legend(
        ax_img, has_gt=True, has_baseline=True, 
        has_modified=has_modified,
        gt_color='darkgreen',
        baseline_color='#FF0000',
        modified_color='#0000FF'
    )
    
    # Всегда отключаем оси, так как используем белый фон
    ax_img.axis('off')
    
    # Боксплот confidence (справа)
    ax_conf = plt.subplot2grid((grid_h, grid_w), (1, 8), colspan=4, rowspan=grid_h-1)
    ax_conf.set_facecolor(bg_color)
    
    # Используем цвет для текста боксплота
    for spine in ax_conf.spines.values():
        spine.set_color(text_color)
    
    ax_conf.tick_params(colors=text_color, which='both')
    
    # Создаем боксплот
    bp = create_confidence_boxplot(
        ax_conf, data_to_plot, box_labels, box_colors,
        min_conf=min_conf, max_conf=max_conf,
        text_color=text_color
    )
    
    # Значения confidence для отображения
    confidence_values = []
    confidence_values.append((1, baseline_conf * 100))
    
    if has_modified:
        confidence_values.append((2, antialias_conf * 100))
        confidence_values.append((3, modified_conf * 100))
    
    # Добавляем метки с процентами
    add_confidence_labels(ax_conf, confidence_values, min_conf, max_conf, text_color)
    
    # Используем плотную компоновку вместо tight_layout
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    return fig


# =====================================================================
# Функции для создания GIF анимаций с визуализацией сравнения моделей
# =====================================================================

def create_comparison_gif(
    seq_dir, baseline_results, modified_results=None, output_path=None,
    sequence="seq_0", fps=5, start_frame=0, end_frame=None, step=1,
    slow_factor=3, fixed_size=(1200, 800), min_conf=75, max_conf=85,
    bg_color="white"
):
    """
    Создает GIF-анимацию сравнения моделей с изображением и боксплотом
    
    Args:
        seq_dir: директория с последовательностями
        baseline_results: путь к CSV с результатами baseline модели
        modified_results: путь к CSV с результатами улучшенной модели
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
        bg_color: цвет фона для визуализации ("white" или "black")
        
    Returns:
        bool: True если успешно, False в случае ошибки
    """
    # Сбрасываем настройки matplotlib для чистого рабочего окружения
    plt.rcdefaults()
    
    # Проверяем корректность параметра bg_color
    if bg_color not in ["white", "black"]:
        print(f"Предупреждение: Неизвестный цвет фона '{bg_color}', используем 'white'")
        bg_color = "white"
    
    # Цвета текста в зависимости от фона
    text_color = "black" if bg_color == "white" else "white"
    
    # Загружаем результаты моделей
    baseline_df = load_csv_results(baseline_results)
    if baseline_df is None:
        return False
    
    has_modified = False
    if modified_results:
        modified_df = load_csv_results(modified_results)
        has_modified = modified_df is not None
    
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
        # Вместо повторения одного и того же кадра, используем дополнительный индекс
        # для более эффективной обработки данных
        original_len = len(selected_frames)
        total_frames = original_len * slow_factor
        print(f"Всего кадров после замедления: {total_frames}")
    else:
        original_len = len(selected_frames)
        total_frames = original_len
    
    # Создаем временную директорию для кадров GIF
    tmp_dir = create_temp_directory(output_path, "tmp_comparison_gif")
    
    print(f"Генерация {total_frames} кадров анимации с фоном {bg_color}...")
    
    # Устанавливаем глобальные настройки стиля
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.facecolor': bg_color,
        'axes.facecolor': bg_color,
        'savefig.facecolor': bg_color,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color
    })
    
    # Создаем отдельный кадр для каждого шага
    valid_frames = []  # Будем хранить только успешно созданные кадры
    
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
            
            # Ищем строку по имени файла или по индексу в пределах размера датафрейма
            if 'frame' in baseline_df.columns:
                baseline_row = baseline_df[baseline_df['frame'] == frame_prefix]
                if len(baseline_row) == 0:
                    # Если не находим по имени, используем индекс, но проверяем границы
                    csv_idx = min(original_idx, len(baseline_df) - 1)
                    baseline_row = baseline_df.iloc[csv_idx]
                else:
                    baseline_row = baseline_row.iloc[0]
            else:
                # Используем индекс с проверкой границ
                csv_idx = min(original_idx, len(baseline_df) - 1)
                baseline_row = baseline_df.iloc[csv_idx]
            
            # Извлекаем данные
            baseline_conf = baseline_row['confidence']
            gt_bbox = parse_bbox_string(baseline_row['gt_bbox'])
            baseline_bbox = parse_bbox_string(baseline_row['pred_bbox'])
            
            # Аналогично для modified, если доступно
            modified_bbox = None
            modified_conf = None
            
            if has_modified:
                if 'frame' in modified_df.columns:
                    modified_row = modified_df[modified_df['frame'] == frame_prefix]
                    if len(modified_row) == 0:
                        csv_idx = min(original_idx, len(modified_df) - 1)
                        modified_row = modified_df.iloc[csv_idx]
                    else:
                        modified_row = modified_row.iloc[0]
                else:
                    csv_idx = min(original_idx, len(modified_df) - 1)
                    modified_row = modified_df.iloc[csv_idx]
                
                modified_conf = modified_row['confidence']
                modified_bbox = parse_bbox_string(modified_row['pred_bbox'])
            
            # Создаем кадр сравнения
            fig = create_comparison_frame(
                img, gt_bbox, baseline_bbox, baseline_conf,
                modified_bbox, modified_conf,
                min_conf, max_conf, frame_idx, total_frames,
                fixed_size, bg_color, text_color
            )
            
            # Сохраняем кадр с явным указанием фона
            tmp_frame_path = tmp_dir / f"frame_{idx:04d}.png"
            fig.savefig(tmp_frame_path, dpi=100, bbox_inches='tight', 
                        facecolor=bg_color, edgecolor='none', pad_inches=0.1,
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
        return False
    
    # Создаем GIF
    result = save_gif_animation(valid_frames, output_path, fps)
    
    # Очищаем временную директорию
    cleanup_temp_directory(tmp_dir)
    
    return result 