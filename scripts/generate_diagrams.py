import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
from PIL import Image

# Создание директории для сохранения изображений
OUTPUT_DIR = "../diploma_pdf/Dissertation/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_receptive_field_diagram():
    """Создает диаграмму рецептивного поля для CNN."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Входное изображение
    input_img = np.zeros((8, 8))
    input_img[3:5, 3:5] = 1  # Выделение центральной области
    
    # Слои CNN с увеличивающимся рецептивным полем
    ax.imshow(input_img, cmap='gray', extent=[0, 8, 0, 8])
    
    # Рецептивные поля разных слоев
    colors = ['red', 'green', 'blue']
    sizes = [1, 3, 5]
    labels = ['Layer 1 (RF=1x1)', 'Layer 2 (RF=3x3)', 'Layer 3 (RF=5x5)']
    
    for i, (color, size, label) in enumerate(zip(colors, sizes, labels)):
        center = 4
        start = center - size // 2
        rect = Rectangle((start, start), size, size, linewidth=2, 
                         edgecolor=color, facecolor='none', label=label)
        ax.add_patch(rect)
    
    ax.set_title('Receptive Field Growth in CNN Layers')
    ax.legend(loc='upper right')
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_yticks(np.arange(0, 9, 1))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'receptive_field_diagram.png'), dpi=300)
    plt.close()
    print("Receptive field diagram created.")

def create_vgg_architecture():
    """Создает схему архитектуры VGG16."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Определение компонентов архитектуры
    blocks = [
        ('Input\n224x224x3', 0),
        ('2x Conv3-64\nMaxPool/2', 1),
        ('2x Conv3-128\nMaxPool/2', 2),
        ('3x Conv3-256\nMaxPool/2', 3),
        ('3x Conv3-512\nMaxPool/2', 4),
        ('3x Conv3-512\nMaxPool/2', 5),
        ('FC-4096\nFC-4096\nFC-1000', 6),
        ('Softmax', 7)
    ]
    
    # Размещение блоков на диаграмме
    for i, (label, pos) in enumerate(blocks):
        ax.add_patch(Rectangle((pos, 0), 0.8, 1, facecolor='lightblue', edgecolor='black'))
        ax.text(pos + 0.4, 0.5, label, ha='center', va='center', fontsize=9)
    
    # Добавление стрелок между блоками
    for i in range(len(blocks) - 1):
        ax.arrow(blocks[i][1] + 0.8, 0.5, blocks[i+1][1] - blocks[i][1] - 0.8, 0, 
                 head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title('VGG16 Architecture')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vgg16_architecture.png'), dpi=300)
    plt.close()
    print("VGG16 architecture diagram created.")

def create_resnet_architecture():
    """Создает схему архитектуры ResNet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
    
    # Основная архитектура ResNet
    blocks = [
        ('Input\n224x224x3', 0),
        ('Conv7\nMaxPool', 1),
        ('3x\nRes Block\n64', 2),
        ('4x\nRes Block\n128', 3),
        ('6x\nRes Block\n256', 4),
        ('3x\nRes Block\n512', 5),
        ('AvgPool\nFC-1000\nSoftmax', 6)
    ]
    
    # Размещение блоков на диаграмме
    for label, pos in blocks:
        ax1.add_patch(Rectangle((pos, 0), 0.8, 1, facecolor='lightgreen', edgecolor='black'))
        ax1.text(pos + 0.4, 0.5, label, ha='center', va='center', fontsize=9)
    
    # Добавление стрелок между блоками
    for i in range(len(blocks) - 1):
        ax1.arrow(blocks[i][1] + 0.8, 0.5, blocks[i+1][1] - blocks[i][1] - 0.8, 0, 
                 head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Детальное изображение остаточного блока
    ax2.add_patch(Rectangle((0, 0.25), 1, 0.5, facecolor='lightgreen', edgecolor='black'))
    ax2.text(0.5, 0.5, 'Conv1x1\nConv3x3\nConv1x1', ha='center', va='center', fontsize=9)
    
    # Стрелка обходного соединения
    ax2.arrow(0, 0.8, 1, 0, head_width=0.05, head_length=0.1, fc='red', ec='red')
    ax2.text(0.5, 0.9, 'Skip Connection', ha='center', va='center', color='red', fontsize=8)
    
    # Сложение
    ax2.add_patch(Rectangle((0.4, 0), 0.2, 0.2, facecolor='yellow', edgecolor='black'))
    ax2.text(0.5, 0.1, '+', ha='center', va='center', fontsize=12)
    
    # Входящие стрелки к сложению
    ax2.arrow(0.5, 0.25, 0, -0.05, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax2.arrow(1, 0.8, -0.5, -0.6, head_width=0.05, head_length=0.05, fc='red', ec='red')
    
    ax1.set_xlim(-0.5, 7)
    ax1.set_ylim(-0.1, 1.1)
    ax1.axis('off')
    ax1.set_title('ResNet-50 Architecture')
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axis('off')
    ax2.set_title('Residual Block')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'resnet_architecture.png'), dpi=300)
    plt.close()
    print("ResNet architecture diagram created.")

def create_yolo_architecture():
    """Создает схему архитектуры YOLOv5."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Основные компоненты YOLOv5
    components = [
        ('Input\n640x640x3', 0, 0.5, 'lightgray'),
        ('Backbone\nCSPDarknet', 2, 0.5, 'lightblue'),
        ('Neck\nFPN+PANet', 4, 0.5, 'lightgreen'),
        ('Head\nDetection', 6, 0.5, 'salmon'),
        ('Output\nBoxes+Classes', 8, 0.5, 'lightgray')
    ]
    
    # Размещение компонентов
    for label, x, y, color in components:
        ax.add_patch(Rectangle((x-0.8, y-0.3), 1.6, 0.6, facecolor=color, edgecolor='black'))
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    
    # Стрелки между компонентами
    for i in range(len(components) - 1):
        ax.arrow(components[i][1] + 0.8, components[i][2], 
                components[i+1][1] - components[i][1] - 1.6, 0, 
                head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Многомасштабные выходы FPN
    scales = [('P5\n80x80', 4, 1.5), ('P4\n40x40', 4, 0), ('P3\n20x20', 4, -1.5)]
    for label, x, y in scales:
        ax.add_patch(Rectangle((x-0.4, y-0.2), 0.8, 0.4, facecolor='palegreen', edgecolor='black'))
        ax.text(x, y, label, ha='center', va='center', fontsize=8)
        ax.arrow(x, y + 0.2, 0, 0.8 - y - 0.2, head_width=0.05, head_length=0.1, 
                fc='black', ec='black', linestyle='--')
        ax.arrow(x, y, 0, -y, head_width=0.05, head_length=0.1, 
                fc='black', ec='black', linestyle='--')
    
    ax.set_xlim(-1, 10)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('YOLOv5 Architecture')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'yolov5_architecture.png'), dpi=300)
    plt.close()
    print("YOLOv5 architecture diagram created.")

def create_aliasing_example():
    """Создает пример алиасинга при даунсэмплинге."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Создание синусоидального сигнала
    x = np.linspace(0, 4*np.pi, 1000)
    y_orig = np.sin(5*x) + np.sin(10*x)
    
    # Исходный сигнал
    axes[0].plot(x, y_orig)
    axes[0].set_title('Original Signal')
    axes[0].set_xlim(0, 4*np.pi)
    axes[0].set_ylim(-2.5, 2.5)
    
    # Корректный даунсэмплинг с фильтрацией
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 0.125)
    y_filtered = filtfilt(b, a, y_orig)
    
    # Даунсэмплинг в 4 раза
    x_down = x[::4]
    y_filtered_down = y_filtered[::4]
    
    axes[1].plot(x, y_orig, 'k-', alpha=0.3)
    axes[1].plot(x, y_filtered, 'g-')
    axes[1].plot(x_down, y_filtered_down, 'ro-')
    axes[1].set_title('Proper Downsampling with Filtering')
    axes[1].set_xlim(0, 4*np.pi)
    axes[1].set_ylim(-2.5, 2.5)
    
    # Некорректный даунсэмплинг без фильтрации
    y_down_aliased = y_orig[::4]
    
    axes[2].plot(x, y_orig, 'k-', alpha=0.3)
    axes[2].plot(x_down, y_down_aliased, 'ro-')
    axes[2].set_title('Aliased Downsampling without Filtering')
    axes[2].set_xlim(0, 4*np.pi)
    axes[2].set_ylim(-2.5, 2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'aliasing_example.png'), dpi=300)
    plt.close()
    print("Aliasing example created.")

def create_blurpool_illustration():
    """Создает иллюстрацию принципа работы BlurPool."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Стандартный максимальный пулинг
    components1 = [
        ('Feature Map\n8x8', 0, 0.5),
        ('Max Pooling\nstride=2', 2, 0.5),
        ('Downsampled\n4x4', 4, 0.5)
    ]
    
    for label, x, y in components1:
        ax1.add_patch(Rectangle((x-0.8, y-0.3), 1.6, 0.6, facecolor='lightblue', edgecolor='black'))
        ax1.text(x, y, label, ha='center', va='center', fontsize=10)
    
    # Стрелки между компонентами
    for i in range(len(components1) - 1):
        ax1.arrow(components1[i][1] + 0.8, components1[i][2], 
                components1[i+1][1] - components1[i][1] - 1.6, 0, 
                head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    ax1.set_title('Standard Max Pooling')
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # BlurPool
    components2 = [
        ('Feature Map\n8x8', 0, 0.5),
        ('Max Pooling\nstride=1', 1.5, 0.5),
        ('Blur Filter\n[1,2,1]', 3, 0.5),
        ('Downsampling\nstride=2', 4.5, 0.5),
        ('Anti-aliased\n4x4', 6, 0.5)
    ]
    
    for label, x, y in components2:
        ax2.add_patch(Rectangle((x-0.7, y-0.3), 1.4, 0.6, facecolor='lightgreen', edgecolor='black'))
        ax2.text(x, y, label, ha='center', va='center', fontsize=9)
    
    # Стрелки между компонентами
    for i in range(len(components2) - 1):
        ax2.arrow(components2[i][1] + 0.7, components2[i][2], 
                components2[i+1][1] - components2[i][1] - 1.4, 0, 
                head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    ax2.set_title('BlurPool')
    ax2.set_xlim(-1, 7)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'blurpool_illustration.png'), dpi=300)
    plt.close()
    print("BlurPool illustration created.")

def create_tips_illustration():
    """Создает иллюстрацию принципа работы TIPS."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Входная карта признаков
    ax.add_patch(Rectangle((0, 3), 2, 2, facecolor='lightblue', edgecolor='black'))
    ax.text(1, 4, 'Input Feature Map\n4x4', ha='center', va='center', fontsize=10)
    
    # Полифазное разбиение (для шага 2)
    polyphase_coords = [(4, 5), (6, 5), (4, 3), (6, 3)]
    polyphase_labels = ['X₀₀', 'X₀₁', 'X₁₀', 'X₁₁']
    
    for i, ((x, y), label) in enumerate(zip(polyphase_coords, polyphase_labels)):
        ax.add_patch(Rectangle((x-0.7, y-0.7), 1.4, 1.4, facecolor='lightgreen', edgecolor='black'))
        ax.text(x, y, f'{label}\n2x2', ha='center', va='center', fontsize=10)
        
        # Стрелки от входной карты к полифазным компонентам
        ax.arrow(2, 4, x - 2 - 0.7, y - 4, head_width=0.1, head_length=0.2, 
                fc='black', ec='black')
    
    # Фильтры для каждой полифазной компоненты
    filter_coords = [(4, 1), (6, 1), (4, -1), (6, -1)]
    filter_labels = ['W₀₀', 'W₀₁', 'W₁₀', 'W₁₁']
    
    for i, ((x, y), label) in enumerate(zip(filter_coords, filter_labels)):
        ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor='pink', edgecolor='black'))
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
        
        # Стрелки от полифазных компонент к фильтрам
        ax.arrow(x, polyphase_coords[i][1] - 0.7, 0, y - polyphase_coords[i][1] + 0.7 - 0.5, 
                head_width=0.1, head_length=0.2, fc='black', ec='black')
    
    # Результаты свертки каждой компоненты
    result_coords = [(10, 5), (12, 5), (10, 3), (12, 3)]
    result_labels = ['Y₀₀', 'Y₀₁', 'Y₁₀', 'Y₁₁']
    
    for i, ((x, y), label) in enumerate(zip(result_coords, result_labels)):
        ax.add_patch(Rectangle((x-0.7, y-0.7), 1.4, 1.4, facecolor='lightyellow', edgecolor='black'))
        ax.text(x, y, f'{label}\n2x2', ha='center', va='center', fontsize=10)
        
        # Стрелки от входной карты к полифазным компонентам
        ax.arrow(polyphase_coords[i][0] + 0.7, polyphase_coords[i][1], 
                x - polyphase_coords[i][0] - 0.7 - 0.7, 0, 
                head_width=0.1, head_length=0.2, fc='black', ec='black')
        
        # Обозначение свертки над стрелкой
        mid_x = (polyphase_coords[i][0] + 0.7 + x - 0.7) / 2
        ax.text(mid_x, y + 0.3, f'* {filter_labels[i]}', ha='center', va='center', fontsize=8)
    
    # Финальное объединение
    ax.add_patch(Rectangle((14, 3), 2, 2, facecolor='gold', edgecolor='black'))
    ax.text(15, 4, 'Output Feature Map\n2x2', ha='center', va='center', fontsize=10)
    
    # Стрелки от результатов к финальному выходу
    for i, (x, y) in enumerate(result_coords):
        ax.arrow(x + 0.7, y, 14 - x - 0.7, 4 - y, head_width=0.1, head_length=0.2, 
                fc='black', ec='black', alpha=0.7)
    
    # Сумма с весами
    ax.text(13, 4.5, 'Σ αᵢⱼYᵢⱼ', ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-1, 17)
    ax.set_ylim(-2, 6)
    ax.axis('off')
    ax.set_title('TIPS (Translation Invariant Polyphase Sampling)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tips_illustration.png'), dpi=300)
    plt.close()
    print("TIPS illustration created.")

if __name__ == "__main__":
    print("Generating diagrams for theoretical section...")
    create_receptive_field_diagram()
    create_vgg_architecture()
    create_resnet_architecture()
    create_yolo_architecture()
    create_aliasing_example()
    create_blurpool_illustration()
    create_tips_illustration()
    print("All diagrams have been created successfully!")
