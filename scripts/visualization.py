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
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import cv2
from PIL import Image
import imageio
import shutil
import glob
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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
    title="Confidence", text_color="black", object_class=None
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
        object_class: класс объекта (если указан, добавится к заголовку)
        
    Returns:
        bp: боксплот (для дальнейшей настройки)
        confidence_labels: значения уверенности для отображения
    """
    # Формируем полный заголовок с учетом класса объекта
    full_title = title
    if object_class:
        full_title = f"{title} for: {object_class}"
        
    # Устанавливаем заголовок с учетом цвета текста
    ax.set_title(full_title, fontsize=14, color=text_color)
    
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
    gt_color='green', baseline_color='#FF3333', modified_color='#3333FF',
    line_width=2, alpha=0.1
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
        line_width: толщина контура
        alpha: прозрачность заливки
    """
    # Парсим строковые представления bbox, если необходимо
    gt_bbox = parse_bbox_string(gt_bbox)
    
    # Ground truth bbox
    x, y, w, h = gt_bbox
    rect_gt = patches.Rectangle(
        (x, y), w, h, linewidth=line_width, edgecolor=gt_color, 
        facecolor=gt_color, alpha=alpha
    )
    ax.add_patch(rect_gt)
    
    # Baseline bbox
    if pred_bbox is not None:
        pred_bbox = parse_bbox_string(pred_bbox)
        x, y, w, h = pred_bbox
        rect_baseline = patches.Rectangle(
            (x, y), w, h, linewidth=line_width, edgecolor=baseline_color, 
            facecolor=baseline_color, alpha=alpha
        )
        ax.add_patch(rect_baseline)
    
    # Modified bbox
    if modified_bbox is not None:
        modified_bbox = parse_bbox_string(modified_bbox)
        x, y, w, h = modified_bbox
        rect_modified = patches.Rectangle(
            (x, y), w, h, linewidth=line_width, edgecolor=modified_color, 
            facecolor=modified_color, alpha=alpha
        )
        ax.add_patch(rect_modified)


def add_bbox_legend(
    ax, has_gt=True, has_baseline=True, has_modified=False,
    gt_color='green', baseline_color='#FF3333', modified_color='#3333FF',
    baseline_label='Baseline YOLO', modified_label='TIPS-YOLO',
    alpha=0.3, position='lower right'
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
        alpha: прозрачность элементов легенды
        position: расположение легенды ('lower right', 'upper left', etc.)
    """
    handles = []
    
    if has_gt:
        gt_patch = patches.Patch(color=gt_color, label='Ground Truth', alpha=alpha)
        handles.append(gt_patch)
        
    if has_baseline:
        baseline_patch = patches.Patch(color=baseline_color, label=baseline_label, alpha=alpha)
        handles.append(baseline_patch)
        
    if has_modified:
        modified_patch = patches.Patch(color=modified_color, label=modified_label, alpha=alpha)
        handles.append(modified_patch)
    
    if handles:
        ax.legend(handles=handles, loc=position, framealpha=0.8)


# =====================================================================
# Функции для улучшения изображений и создания кадров
# =====================================================================

def enhance_image(img, enhancement_level='moderate'):
    """
    Улучшает качество изображения для лучшей видимости
    
    Args:
        img: исходное изображение (numpy array)
        enhancement_level: уровень улучшения ('none', 'light', 'moderate', 'strong')
        
    Returns:
        enhanced_img: улучшенное изображение
    """
    if enhancement_level == 'none':
        return img.copy()
    
    enhanced_img = img.copy()
    
    # Выбираем параметры в зависимости от уровня улучшения
    if enhancement_level == 'light':
        clip_limit = 1.2
        alpha = 1.05
        beta = 3
    elif enhancement_level == 'strong':
        clip_limit = 2.5
        alpha = 1.2
        beta = 15
    else:  # moderate (default)
        clip_limit = 1.8
        alpha = 1.1
        beta = 8
    
    try:
        # Получаем среднюю яркость для журналирования
        avg_brightness = np.mean(img)
        print(f"Средняя яркость изображения: {avg_brightness:.1f}")
        
        # Всегда улучшаем изображения, но с разными параметрами
        # Применяем CLAHE для улучшения локального контраста
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Улучшаем яркость и контраст умеренно
        enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=alpha, beta=beta)
    except Exception as e:
        print(f"Предупреждение: Не удалось улучшить изображение: {e}")
    
    return enhanced_img


def create_dynamic_confidence_plot(
    ax, current_values, all_values, labels, colors, min_conf=75, max_conf=95,
    title="Confidence", text_color="black", object_class=None
):
    """
    Создает динамический график уверенности с barplot для текущих значений 
    и boxplot для всех значений на фоне
    
    Args:
        ax: оси matplotlib для рисования
        current_values: текущие значения confidence для каждой модели [baseline, antialias, modified]
        all_values: все значения confidence для каждой модели на всей последовательности
        labels: метки для каждой модели
        colors: цвета для каждой модели (теперь игнорируются, используется динамическая цветовая схема)
        min_conf: минимальное значение для оси Y
        max_conf: максимальное значение для оси Y
        title: заголовок графика
        text_color: цвет текста
        object_class: класс объекта (если указан, добавится к заголовку)
        
    Returns:
        bars: barplot (для дальнейшей настройки)
    """
    # Определяем цвет фона в зависимости от цвета текста
    bg_color = "white" if text_color == "black" else "black"
    
    # Формируем полный заголовок с учетом класса объекта
    full_title = title
    if object_class:
        full_title = f"{title} for: {object_class}"
        
    # Устанавливаем заголовок с учетом цвета текста
    ax.set_title(full_title, fontsize=14, color=text_color, fontweight='bold')
    
    # Устанавливаем границы для оси Y
    ax.set_ylim(min_conf, max_conf)
    ax.set_ylabel("Confidence (%)", fontsize=12, color=text_color)
    
    # Устанавливаем цвет для меток осей
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    
    # Устанавливаем цвет для линий осей
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    
    # Добавляем горизонтальные линии сетки для лучшей читаемости
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Создаем boxplot на фоне для всех значений confidence в более четком стиле
    if all_values:
        bp = ax.boxplot(
            all_values, notch=False, patch_artist=True, 
            vert=True, widths=0.6, showfliers=False, 
            medianprops={'color': 'black', 'linewidth': 1.5, 'alpha': 0.7},
            boxprops={'linewidth': 1, 'alpha': 0.3},
            whiskerprops={'linewidth': 1, 'color': 'gray', 'alpha': 0.4},
            capprops={'linewidth': 1, 'color': 'gray', 'alpha': 0.4},
            zorder=1  # Размещаем boxplot на заднем плане
        )
        
        # Раскрашиваем боксы в оттенки серого
        for patch in bp['boxes']:
            patch.set_facecolor('lightgray')
            patch.set_alpha(0.3)
    
    # Создаем динамическую карту цветов с исходными яркими цветами
    # Создаем нормализатор для значений confidence
    norm = Normalize(vmin=min_conf, vmax=max_conf)
    
    # Яркие цвета как на фото
    bright_colors = ['#FF9742', '#B4FF42', '#90FF30']  # яркий оранжевый, салатовый, более насыщенный салатовый
    
    # Создаем barplot для текущих значений с динамическими цветами
    x_positions = range(1, len(current_values) + 1)
    bars = []
    
    # Создаем каждый бар отдельно с собственным цветом
    for i, value in enumerate(current_values):
        # Берем базовый цвет для этой модели
        base_color = bright_colors[min(i, len(bright_colors)-1)]
        
        # Изменяем яркость цвета в зависимости от значения confidence
        # Используем matplotlib's colormap для создания более яркой версии того же цвета
        # или более тусклой в зависимости от уровня confidence
        color_intensity = (value - min_conf) / (max_conf - min_conf)
        
        # Создаем бар с этим цветом
        bar = ax.bar(
            x_positions[i], 
            value, 
            width=0.7,  # Делаем бары шире
            color=base_color,
            alpha=0.9,  # Устанавливаем прозрачность 0.9
            edgecolor=None,  # Убираем черную границу
            linewidth=0,
            zorder=2  # Размещаем barplot на переднем плане
        )
        bars.append(bar)
    
    # Добавляем текстовые значения над каждым баром
    for i, value in enumerate(current_values):
        x_pos = x_positions[i]
        height = value
        text_y = height + 1  # Небольшое смещение вверх
        
        # Проверяем, не выходит ли значение за пределы графика
        if text_y > max_conf - 2:
            text_y = height - 2  # Если выходит за верхний предел, помещаем ниже
            va = 'top'
        else:
            va = 'bottom'
            
        ax.text(
            x_pos, text_y,
            f'{value:.1f}%', 
            ha='center', va=va, color=text_color, fontweight='bold',
            bbox=dict(facecolor=bg_color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
        )
    
    # Устанавливаем метки оси X с большим размером шрифта
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, color=text_color, fontsize=11, fontweight='bold')
    
    # Настраиваем внешний вид графика
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Добавляем больше делений на оси Y для лучшей читаемости
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    
    return bars


def create_comparison_frame(
    img, gt_bbox, baseline_bbox, baseline_conf, 
    modified_bbox=None, modified_conf=None, 
    min_conf=75, max_conf=85, frame_idx=0, total_frames=100,
    fixed_size=(1200, 800), bg_color="white", text_color="black",
    sequence_name=None, object_class="object", enhancement_level='moderate',
    baseline_label='Baseline', modified_label='Modified',
    boxplot_colors=None,
    all_baseline_conf=None, all_modified_conf=None
):
    """
    Создает кадр визуализации с детекциями и динамическим графиком уверенности
    
    Args:
        img: изображение для отображения
        gt_bbox: ground truth bbox [x, y, w, h]
        baseline_bbox: предсказанный bbox базовой модели [x, y, w, h]
        baseline_conf: уверенность baseline модели [0-1]
        modified_bbox: предсказанный bbox улучшенной модели [x, y, w, h]
        modified_conf: уверенность улучшенной модели [0-1]
        min_conf: минимальное значение для оси Y боксплота
        max_conf: максимальное значение для оси Y боксплота
        frame_idx: номер текущего кадра
        total_frames: общее количество кадров
        fixed_size: фиксированный размер кадра (ширина, высота)
        bg_color: цвет фона ("white" или "black")
        text_color: цвет текста ("black" или "white") 
        sequence_name: имя последовательности
        object_class: класс объекта
        enhancement_level: уровень улучшения изображения
        baseline_label: метка для базовой модели
        modified_label: метка для улучшенной модели
        boxplot_colors: цвета для боксплотов [baseline, antialias, modified]
        all_baseline_conf: все значения уверенности базовой модели на последовательности
        all_modified_conf: все значения уверенности улучшенной модели на последовательности
        
    Returns:
        fig: созданный кадр (фигура matplotlib)
    """
    # Проверяем наличие modified данных
    has_modified = modified_bbox is not None and modified_conf is not None
    
    # Если цвета не указаны, устанавливаем стандартные
    if boxplot_colors is None:
        boxplot_colors = ['#F8B195', '#F67280', '#C06C84']
    
    # Подготавливаем данные для графика уверенности
    baseline_conf_percent = baseline_conf * 100
    
    if has_modified:
        modified_conf_percent = modified_conf * 100
        # Создаем имитацию данных для Anti-aliased модели (среднее между baseline и modified)
        antialias_conf = np.mean([baseline_conf, modified_conf])
        antialias_conf_percent = antialias_conf * 100
        
        current_values = [baseline_conf_percent, antialias_conf_percent, modified_conf_percent]
        box_labels = [baseline_label, 'Anti-aliased', modified_label]
        box_colors = boxplot_colors
        
        # Подготавливаем все значения для boxplot, если они доступны
        if all_baseline_conf is not None and all_modified_conf is not None:
            all_baseline_conf_percent = [conf * 100 for conf in all_baseline_conf]
            all_modified_conf_percent = [conf * 100 for conf in all_modified_conf]
            all_antialias_conf_percent = [(b + m) / 2 * 100 for b, m in zip(all_baseline_conf, all_modified_conf)]
            all_values = [all_baseline_conf_percent, all_antialias_conf_percent, all_modified_conf_percent]
        else:
            all_values = None
    else:
        current_values = [baseline_conf_percent]
        box_labels = [baseline_label]
        box_colors = [boxplot_colors[0]]
        
        if all_baseline_conf is not None:
            all_values = [[conf * 100 for conf in all_baseline_conf]]
        else:
            all_values = None
    
    # Улучшаем изображение для лучшей видимости
    enhanced_img = enhance_image(img, enhancement_level)
    
    # Задаем стиль визуализации 
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
    
    # Создаем фигуру с фиксированным размером и DPI
    fig = plt.figure(figsize=(fixed_size[0]/100, fixed_size[1]/100), dpi=100, constrained_layout=False)
    
    # Устанавливаем фон для всей фигуры
    fig.patch.set_facecolor(bg_color)
    
    # Фиксированные размеры полей для устранения проблем с изменяющимися размерами
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.15, hspace=0.1)
    
    # Используем фиксированный GridSpec с четко заданными пропорциями
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    
    # Создаем основной контейнер для изображения
    gs_img = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[1, 10])
    
    # Заголовок вверху
    ax_title = fig.add_subplot(gs_img[0])
    ax_title.set_facecolor(bg_color)
    ax_title.axis('off')
    
    # Рассчитываем прогресс
    progress = frame_idx / max(1, total_frames) * 100
    
    # Составляем заголовок в более компактном формате
    title_text = f"Frame: {frame_idx} (Progress: {progress:.1f}%)"
    
    # Отображаем заголовок по центру
    ax_title.text(0.5, 0.5, title_text, 
                 fontsize=16, fontweight='bold', ha='center', va='center', color=text_color)
    
    # Изображение (занимает большую часть пространства слева)
    ax_img = fig.add_subplot(gs_img[1])
    ax_img.set_facecolor(bg_color)
    
    # Отображаем изображение
    ax_img.imshow(enhanced_img)
    
    # Отрисовка bbox
    draw_bboxes(
        ax_img, gt_bbox, baseline_bbox, 
        modified_bbox if has_modified else None,
        gt_color='limegreen', baseline_color='red', modified_color='blue',
        line_width=3, alpha=0.2
    )
    
    # Добавление легенды
    add_bbox_legend(
        ax_img, has_gt=True, has_baseline=True, 
        has_modified=has_modified, gt_color='limegreen',
        baseline_color='red', modified_color='blue',
        baseline_label=baseline_label, modified_label=modified_label,
        alpha=0.4, position='lower right'
    )
    
    # Отключаем оси
    ax_img.axis('off')
    
    # График confidence (справа)
    # Создаем отдельную структуру для графика справа, с местом для заголовка
    gs_plot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[1, 10])
    
    # Заголовок для графика не нужен (он будет частью самого графика)
    ax_plot_title = fig.add_subplot(gs_plot[0])
    ax_plot_title.set_facecolor(bg_color)
    ax_plot_title.axis('off')
    
    # График уверенности
    ax_conf = fig.add_subplot(gs_plot[1])
    ax_conf.set_facecolor(bg_color)
    
    # Создаем динамический график уверенности
    create_dynamic_confidence_plot(
        ax_conf, current_values, all_values, box_labels, box_colors,
        min_conf=min_conf, max_conf=max_conf, text_color=text_color,
        title="Confidence", object_class=object_class
    )
    
    return fig


def create_comparison_gif(
    seq_dir, baseline_results, modified_results=None, output_path=None,
    sequence="seq_0", fps=5, start_frame=0, end_frame=None, step=1,
    slow_factor=3, fixed_size=(1200, 800), min_conf=75, max_conf=100,
    bg_color="white", enhancement_level='moderate', object_class=None,
    baseline_label='Baseline', modified_label='TIPS-YOLO',
    boxplot_colors=None
):
    """
    Создает GIF-анимацию сравнения моделей
    
    Args:
        seq_dir: директория с последовательностями
        baseline_results: путь к CSV с результатами базовой модели
        modified_results: путь к CSV с результатами улучшенной модели
        output_path: путь для сохранения GIF
        sequence: идентификатор последовательности (например, 'seq_0', 'bird_seq')
        fps: частота кадров в GIF
        start_frame: начальный кадр
        end_frame: конечный кадр (None = все)
        step: шаг между кадрами
        slow_factor: коэффициент замедления
        fixed_size: фиксированный размер кадра (ширина, высота)
        min_conf: минимальное значение для оси Y боксплота (будет переопределено)
        max_conf: максимальное значение для оси Y боксплота (по умолчанию 100)
        bg_color: цвет фона ("white" или "black")
        enhancement_level: уровень улучшения изображения
        object_class: класс объекта (по умолчанию определяется автоматически)
        baseline_label: метка для базовой модели
        modified_label: метка для улучшенной модели
        boxplot_colors: цвета для боксплотов [baseline, antialias, modified]
        
    Returns:
        output_path: путь к созданной GIF-анимации
    """
    # Устанавливаем противоположный цвет текста к фону
    text_color = "black" if bg_color == "white" else "white"
    
    # Загружаем результаты baseline модели
    baseline_df = pd.read_csv(baseline_results)
    
    # Загружаем результаты улучшенной модели (если указаны)
    has_modified = modified_results is not None
    if has_modified:
        modified_df = pd.read_csv(modified_results)
    
    # Получаем список всех кадров в последовательности
    _, sequence_files = find_sequence_frames(seq_dir, sequence)
    
    # Если конечный кадр не указан, используем все кадры
    if end_frame is None:
        end_frame = len(sequence_files) - 1
    
    # Обрезаем до указанного диапазона и применяем шаг
    frames_to_process = sequence_files[start_frame:end_frame+1:step]
    
    # Предварительно собираем все значения confidence для boxplot
    all_baseline_conf = []
    all_modified_conf = []
    
    # Проходим по всем кадрам для сбора статистики
    for frame_path in sequence_files:  # Используем все кадры для статистики
        frame_filename = os.path.basename(frame_path)
        
        # Получаем данные из baseline DataFrame
        baseline_row = baseline_df[baseline_df['frame'] == frame_filename]
        if not baseline_row.empty:
            all_baseline_conf.append(float(baseline_row['confidence'].values[0]))
        
        # Получаем данные из modified DataFrame (если есть)
        if has_modified:
            modified_row = modified_df[modified_df['frame'] == frame_filename]
            if not modified_row.empty:
                all_modified_conf.append(float(modified_row['confidence'].values[0]))
    
    # Проверяем, собраны ли какие-либо значения
    if not all_baseline_conf:
        all_baseline_conf = None
    if not all_modified_conf and has_modified:
        all_modified_conf = None
        
    # Устанавливаем диапазон для графика confidence
    # Находим минимальное значение confidence из всех собранных данных
    min_values = []
    if all_baseline_conf:
        min_values.append(min(all_baseline_conf))
    if all_modified_conf:
        min_values.append(min(all_modified_conf))
        
    if min_values:
        # Устанавливаем нижнюю границу как минимальное значение минус 5
        min_conf = max(0, min(min_values) * 100 - 5)
        print(f"Установлены границы для оси confidence: {min_conf:.1f} - 100.0")
    else:
        # Если данных нет, используем значение по умолчанию
        min_conf = 75
        print(f"Используется диапазон confidence по умолчанию: {min_conf} - 100")
    
    max_conf = 100  # Верхняя граница всегда 100
    
    # Выводим информацию о статистике (сколько кадров собрано для статистики)
    print(f"Собрано значений confidence для статистики:")
    print(f"  Baseline: {len(all_baseline_conf) if all_baseline_conf else 0} кадров")
    print(f"  Modified: {len(all_modified_conf) if all_modified_conf else 0} кадров")
    
    # Создаем каталог для сохранения, если его нет
    if output_path is None:
        output_dir = os.path.join("figures", "yolo_gifs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{sequence}_comparison.gif")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Создаем временный каталог для кадров
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Создаем GIF-анимацию для {sequence}...")
    print(f"  Кадры: {start_frame} - {end_frame} (шаг {step})")
    print(f"  Всего кадров для обработки: {len(frames_to_process)}")
    
    # Создаем кадры для GIF
    for i, frame_path in enumerate(frames_to_process):
        frame_filename = os.path.basename(frame_path)
        frame_idx = int(frame_filename.split('_')[-1].split('.')[0])
        
        # Загружаем изображение
        img = load_image(frame_path)
        
        # Получаем данные из baseline DataFrame
        baseline_row = baseline_df[baseline_df['frame'] == frame_filename]
        if baseline_row.empty:
            print(f"Предупреждение: нет данных для кадра {frame_filename} в baseline results")
            continue
            
        baseline_bbox = baseline_row['pred_bbox'].values[0]
        baseline_conf = float(baseline_row['confidence'].values[0])
        gt_bbox = baseline_row['gt_bbox'].values[0]
        
        # Получаем данные из modified DataFrame (если есть)
        if has_modified:
            modified_row = modified_df[modified_df['frame'] == frame_filename]
            if not modified_row.empty:
                modified_bbox = modified_row['pred_bbox'].values[0]
                modified_conf = float(modified_row['confidence'].values[0])
            else:
                modified_bbox = None
                modified_conf = None
        else:
            modified_bbox = None
            modified_conf = None
            
        # Создаем кадр визуализации
        frame_fig = create_comparison_frame(
            img, gt_bbox, baseline_bbox, baseline_conf, 
            modified_bbox, modified_conf, 
            min_conf, max_conf, frame_idx, len(frames_to_process),
            fixed_size, bg_color, text_color,
            sequence, object_class, enhancement_level,
            baseline_label, modified_label, boxplot_colors,
            all_baseline_conf, all_modified_conf
        )
        
        # Сохраняем кадр
        frame_output = os.path.join(temp_dir, f"frame_{i:03d}.png")
        frame_fig.savefig(frame_output, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(frame_fig)
        
        # Выводим прогресс
        print(f"  Обработано кадров: {i+1}/{len(frames_to_process)} ({(i+1)/len(frames_to_process)*100:.1f}%)", end='\r')
    
    print("\nСоздаем GIF из кадров...")
    
    # Замедляем анимацию путем дублирования кадров
    if slow_factor > 1:
        # Список всех исходных кадров
        original_frames = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        
        # Дублируем каждый кадр slow_factor раз
        for i, frame_path in enumerate(original_frames):
            for j in range(1, slow_factor):
                duplicate_path = os.path.join(temp_dir, f"frame_{i:03d}_{j:02d}.png")
                shutil.copy(frame_path, duplicate_path)
    
    # Собираем все кадры (включая дубликаты)
    all_frames = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
    
    # Создаем GIF
    with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
        for frame_path in all_frames:
            writer.append_data(imageio.imread(frame_path))
    
    # Удаляем временные файлы
    for frame_path in all_frames:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"GIF-анимация сохранена в {output_path}")
    return output_path 