# Shift-Invariance

Проект для исследования инвариантности к сдвигам нейронных сетей и методов обработки изображений. Фокус на улучшении устойчивости моделей к субпиксельным сдвигам объектов.

## Структура проекта

Проект организован в модульную структуру:

### Основные модули:
- `scripts/` - Основные модули и утилиты проекта
  - `utils.py` - Общие утилиты для работы с изображениями, bbox, метриками и файлами
  - `visualization.py` - Функции для визуализации и создания GIF-анимаций для классификаторов
  - `visualization_yolo.py` - Специализированные функции для визуализации YOLO-моделей
  - `detection.py` - Функции для запуска YOLO-детекторов и обработки результатов
  - `create_visualizations.py` - CLI-интерфейс для создания визуализаций
  - `run_detector.py` - CLI-интерфейс для запуска YOLO-детекторов

### Данные и результаты:
- `data/` - Данные для экспериментов
  - `backgrounds/` - Фоновые изображения для синтетических последовательностей
  - `objects/` - Объекты для наложения на фоны
  - `sequences/` - Сгенерированные последовательности
  - `sequences_smooth/` - Последовательности с плавными движениями

- `sequences/` - Последовательности изображений для тестирования
  - `seq_0/` - Последовательность 0 (полные данные для YOLO и TIPS-YOLO)
  - `seq_1/` - Последовательность 1 (данные только для базовой YOLO)
  - `seq_2/` - Последовательность 2 (данные только для базовой YOLO)
  - `bird_seq/` - Последовательность с птицей

- `checkpoints/` - Сохраненные веса моделей

- `results/` - Результаты экспериментов
  - `classifiers/` - Результаты для классификаторов
  - `yolo/` - Результаты для YOLO-моделей (реальные данные)
  - `yolo_tips/` - Результаты для TIPS-YOLO (реальные данные)

### Визуализации:
- `figures/` - Графики и визуализации
  - `boxplot_gifs/` - GIF-анимации с боксплотами
  - `yolo_gifs/` - GIF-анимации с визуализацией YOLO-детекций
  - `yolo_real_gifs/` - GIF-визуализации на основе реальных результатов (белый фон)

## Использование

### Запуск детекции на последовательностях:

```bash
python -m scripts.run_detector --weights /path/to/yolo_weights.pt --seq-dir data/sequences --out-dir results/yolo
```

### Создание GIF-анимаций сравнения YOLO моделей:

```bash
python -m scripts.create_visualizations yolo-comparison --seq-dir data/sequences --baseline results/yolo/baseline --modified results/yolo_tips/tips --out-dir figures/yolo_gifs
```

### Создание визуализаций на основе реальных данных:

```bash
# Создание GIF-визуализации для seq_0 (есть данные и для YOLO, и для TIPS-YOLO)
python -m scripts.create_visualizations yolo-comparison --seq-dir sequences/seq_0 --baseline results/yolo/baseline_seq_0.csv --modified results/yolo_tips/tips_seq_0.csv --out-dir figures/yolo_gifs

# Создание GIF-визуализации для seq_1 (только данные базовой YOLO)
python -m scripts.create_visualizations yolo-comparison --seq-dir sequences/seq_1 --baseline results/yolo/baseline_seq_1.csv --out-dir figures/yolo_gifs

# Создание GIF-визуализации для seq_2 (только данные базовой YOLO)
python -m scripts.create_visualizations yolo-comparison --seq-dir sequences/seq_2 --baseline results/yolo/baseline_seq_2.csv --out-dir figures/yolo_gifs
```

## ВАЖНАЯ ИНФОРМАЦИЯ: Реальные данные для дипломной работы

Для дипломной работы необходимо использовать только реальные данные:

- **Результаты**: директории `results/yolo/` и `results/yolo_tips/`
- **Визуализации**: директории `figures/yolo_gifs/` 

## Параметры для скриптов визуализации

Скрипт `create_visualizations.py` поддерживает следующие параметры:

- `--seq-dir` - Директория с последовательностями изображений
- `--baseline` - Путь к CSV с результатами базовой YOLO модели
- `--modified` - Путь к CSV с результатами улучшенной YOLO модели (опционально)
- `--out-dir` - Директория для сохранения GIF-анимаций
- `--sequence` - Конкретная последовательность для обработки
- `--fps` - Кадров в секунду в GIF
- `--start-frame` - Начальный кадр
- `--end-frame` - Конечный кадр
- `--step` - Шаг между кадрами
- `--slow-factor` - Коэффициент замедления анимации
- `--width` - Ширина кадров GIF в пикселях
- `--height` - Высота кадров GIF в пикселях
- `--min-conf` - Минимальное значение для оси Y боксплота
- `--max-conf` - Максимальное значение для оси Y боксплота

## Окружение и зависимости

Проект использует следующие библиотеки:
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Pillow
- тензорные фреймворки (для запуска моделей)

Установка зависимостей:
```bash
pip install -r requirements.txt
``` 