#!/usr/bin/env python3
"""
Модуль для запуска классификаторов на последовательностях изображений
и сохранения их выходных данных (penultimate features, классы, логиты).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import cosine

# Импорт утилит из общего модуля
from scripts.utils import find_sequence_ids


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Запуск классификаторов на последовательностях изображений"
    )
    
    # Обязательные аргументы
    parser.add_argument(
        "--seq-dir", type=str, required=True,
        help="Директория с последовательностями"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Директория для сохранения результатов"
    )
    
    # Опциональные аргументы
    parser.add_argument(
        "--models", type=str, default="vgg16,resnet50",
        help="Модели через запятую (vgg16,aa-vgg16,resnet50,aa-resnet50)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Размер батча для инференса"
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints",
        help="Директория с pre-trained весами моделей"
    )
    
    return parser.parse_args()


def load_model(model_name, ckpt_dir, device):
    """
    Загружает модель классификатора по имени
    
    Args:
        model_name: имя модели (vgg16, aa-vgg16, resnet50, aa-resnet50)
        ckpt_dir: директория с весами
        device: устройство для загрузки (cuda/cpu)
        
    Returns:
        model: модель PyTorch
        features_layer: имя слоя для извлечения предпоследних признаков
    """
    model_name = model_name.lower()
    
    if model_name == "vgg16":
        try:
            # Пробуем загрузить модель из локальных весов
            model_path = Path(ckpt_dir) / "vgg16.pth"
            if model_path.exists():
                model = models.vgg16(pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Загружена VGG16 из {model_path}")
            else:
                # Если нет, загружаем pretrained из torchvision
                model = models.vgg16(pretrained=True)
                print("Загружена pretrained VGG16 из torchvision")
                
            features_layer = "classifier.4"  # fc1 после ReLU
            
        except Exception as e:
            print(f"Ошибка при загрузке VGG16: {e}")
            return None, None
    
    elif model_name == "aa-vgg16":
        try:
            # Импортируем антиалиасинговые модели
            import antialiased_cnns
            
            # Используем предобученную модель из библиотеки antialiased-cnns
            model = antialiased_cnns.vgg16(pretrained=True)
            print("Загружена предобученная AA-VGG16 из antialiased-cnns")
            
            features_layer = "classifier.4"
                
        except Exception as e:
            print(f"Ошибка при загрузке AA-VGG16: {e}")
            return None, None
    
    elif model_name == "resnet50":
        try:
            model_path = Path(ckpt_dir) / "resnet50.pth"
            if model_path.exists():
                model = models.resnet50(pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Загружена ResNet50 из {model_path}")
            else:
                model = models.resnet50(pretrained=True)
                print("Загружена pretrained ResNet50 из torchvision")
                
            features_layer = "avgpool"  # предпоследний слой в ResNet
            
        except Exception as e:
            print(f"Ошибка при загрузке ResNet50: {e}")
            return None, None
    
    elif model_name == "aa-resnet50":
        try:
            # Импортируем антиалиасинговые модели
            import antialiased_cnns
            
            # Используем предобученную модель из библиотеки antialiased-cnns
            model = antialiased_cnns.resnet50(pretrained=True)
            print("Загружена предобученная AA-ResNet50 из antialiased-cnns")
                
            features_layer = "avgpool"
            
        except Exception as e:
            print(f"Ошибка при загрузке AA-ResNet50: {e}")
            return None, None
    
    elif model_name == "tips-vgg16":
        try:
            # Для TIPS версии используем обычную VGG16, а затем будем применять сглаживание к выходам
            model = models.vgg16(pretrained=True)
            print("Загружена базовая VGG16 для TIPS адаптации")
            
            features_layer = "classifier.4"
            
        except Exception as e:
            print(f"Ошибка при загрузке TIPS-VGG16: {e}")
            return None, None
    
    elif model_name == "tips-resnet50":
        try:
            # Для TIPS версии используем обычную ResNet50, а затем будем применять сглаживание к выходам
            model = models.resnet50(pretrained=True)
            print("Загружена базовая ResNet50 для TIPS адаптации")
            
            features_layer = "avgpool"
            
        except Exception as e:
            print(f"Ошибка при загрузке TIPS-ResNet50: {e}")
            return None, None
    
    else:
        print(f"Ошибка: Неизвестная модель {model_name}")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, features_layer


def extract_features(model, feature_layer, images, device):
    """
    Извлекает предпоследние признаки из модели
    
    Args:
        model: модель PyTorch
        feature_layer: название слоя для извлечения
        images: батч тензоров изображений
        device: устройство (cuda/cpu)
        
    Returns:
        features: тензор признаков (N, C)
        logits: выходные логиты модели (N, 1000)
    """
    images = images.to(device)
    features = None
    
    # Регистрируем хук для извлечения предпоследних признаков
    def hook_fn(module, input, output):
        nonlocal features
        # Преобразуем в плоский вектор
        if len(output.shape) > 2:
            features = output.view(output.size(0), -1)
        else:
            features = output
    
    # Находим нужный слой в модели
    for name, module in model.named_modules():
        if name == feature_layer:
            handle = module.register_forward_hook(hook_fn)
            break
    
    # Прямой проход для получения выходов модели
    with torch.no_grad():
        logits = model(images)
    
    # Удаляем хук
    handle.remove()
    
    return features.cpu().numpy(), logits.cpu().numpy()


def compute_cosine_similarity(features1, features2):
    """
    Вычисляет косинусное сходство между векторами признаков
    
    Args:
        features1: первый вектор признаков
        features2: второй вектор признаков
        
    Returns:
        float: косинусное сходство (1 - полное сходство, 0 - ортогональность)
    """
    return 1 - cosine(features1, features2)


def run_classifier_on_sequence(
    model, model_name, feature_layer, sequence_dir, sequence_id, 
    transform, batch_size, device, output_dir
):
    """
    Запускает классификатор на последовательности и сохраняет результаты
    
    Args:
        model: модель PyTorch
        model_name: имя модели (для логов и имени файла)
        feature_layer: слой для извлечения признаков
        sequence_dir: путь к директории с последовательностью
        sequence_id: идентификатор последовательности
        transform: преобразования для изображений
        batch_size: размер батча
        device: устройство (cuda/cpu)
        output_dir: директория для сохранения результатов
        
    Returns:
        pandas.DataFrame: результаты для всей последовательности
    """
    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем все изображения в последовательности
    image_files = sorted(list(sequence_dir.glob(f"{sequence_id}_*.png")))
    
    if not image_files:
        print(f"Ошибка: Не найдены изображения для последовательности {sequence_id}")
        return None
    
    results = []
    
    # Определяем, является ли это TIPS моделью
    is_tips_model = model_name.lower().startswith("tips-")
    
    # Для TIPS моделей загружаем и обрабатываем всю последовательность сразу
    if is_tips_model:
        all_images = []
        for img_path in image_files:
            # Загружаем изображение
            img = Image.open(img_path).convert('RGB')
            # Применяем преобразования
            img_tensor = transform(img)
            all_images.append(img_tensor)
        
        # Пакетная обработка всей последовательности
        all_features = []
        all_logits = []
        
        # Обрабатываем изображения пакетами
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            batch_tensor = torch.stack(batch, dim=0)
            
            # Получаем признаки и логиты для текущего пакета
            features, logits = extract_features(model, feature_layer, batch_tensor, device)
            
            all_features.extend(features)
            all_logits.extend(logits)
        
        # Применяем временное сглаживание для TIPS (скользящее среднее)
        window_size = 5  # Размер окна сглаживания
        
        # Применяем сглаживание к признакам
        smoothed_features = []
        for i in range(len(all_features)):
            # Определяем начало и конец окна
            start = max(0, i - window_size // 2)
            end = min(len(all_features), i + window_size // 2 + 1)
            
            # Вычисляем среднее значение признаков в окне
            window_features = np.array(all_features[start:end])
            smoothed_feature = np.mean(window_features, axis=0)
            smoothed_features.append(smoothed_feature)
        
        # Аналогично для логитов (softmax будет применяться после сглаживания)
        smoothed_logits = []
        for i in range(len(all_logits)):
            start = max(0, i - window_size // 2)
            end = min(len(all_logits), i + window_size // 2 + 1)
            
            window_logits = np.array(all_logits[start:end])
            smoothed_logit = np.mean(window_logits, axis=0)
            smoothed_logits.append(smoothed_logit)
        
        # Формируем результаты
        for i, img_path in enumerate(image_files):
            # Получаем номер кадра из имени файла
            frame_num = int(img_path.stem.split('_')[-1])
            
            # Используем сглаженные признаки и логиты
            feature = smoothed_features[i]
            logit = smoothed_logits[i]
            
            # Применяем softmax к логитам
            softmax = np.exp(logit) / np.sum(np.exp(logit))
            # Получаем предсказанный класс и вероятность
            pred_class = np.argmax(softmax)
            confidence = softmax[pred_class]
            
            # Вычисляем косинусное сходство с первым кадром (если мы не на первом кадре)
            if i == 0:
                cos_sim = 1.0  # Для первого кадра сходство с самим собой равно 1
                feature_first = feature  # Запоминаем признаки первого кадра
            else:
                cos_sim = compute_cosine_similarity(feature, feature_first)
                
            # Изменение уверенности относительно первого кадра
            if i == 0:
                conf_drift = 0.0
                confidence_first = confidence
            else:
                conf_drift = abs(confidence - confidence_first)
            
            # Добавляем результаты в список
            result = {
                'frame': frame_num,
                'cos_sim': cos_sim,
                'confidence': confidence,
                'pred_class': int(pred_class),
                'conf_drift': conf_drift
            }
            results.append(result)
    else:
        # Для обычных моделей (не TIPS) используем оригинальный код
        # Обрабатываем пакетами
        feature_first = None  # Признаки первого кадра
        confidence_first = None  # Уверенность для первого кадра
        
        # Обрабатываем изображения пакетами
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_tensors = []
            
            for img_path in batch_files:
                # Загружаем изображение
                img = Image.open(img_path).convert('RGB')
                # Применяем преобразования
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
            
            # Объединяем в батч
            batch_tensor = torch.stack(batch_tensors, dim=0)
            
            # Получаем признаки и логиты
            features, logits = extract_features(model, feature_layer, batch_tensor, device)
            
            # Обрабатываем каждый элемент батча
            for j, img_path in enumerate(batch_files):
                # Получаем номер кадра из имени файла
                frame_num = int(img_path.stem.split('_')[-1])
                
                # Получаем признаки и логиты для текущего изображения
                feature = features[j]
                logit = logits[j]
                
                # Применяем softmax к логитам
                softmax = np.exp(logit) / np.sum(np.exp(logit))
                # Получаем предсказанный класс и вероятность
                pred_class = np.argmax(softmax)
                confidence = softmax[pred_class]
                
                # Вычисляем косинусное сходство с первым кадром (если мы не на первом кадре)
                if feature_first is None:
                    cos_sim = 1.0  # Для первого кадра сходство с самим собой равно 1
                    feature_first = feature  # Запоминаем признаки первого кадра
                else:
                    cos_sim = compute_cosine_similarity(feature, feature_first)
                    
                # Изменение уверенности относительно первого кадра
                if confidence_first is None:
                    conf_drift = 0.0
                    confidence_first = confidence
                else:
                    conf_drift = abs(confidence - confidence_first)
                
                # Добавляем результаты в список
                result = {
                    'frame': frame_num,
                    'cos_sim': cos_sim,
                    'confidence': confidence,
                    'pred_class': int(pred_class),
                    'conf_drift': conf_drift
                }
                results.append(result)
    
    # Создаем DataFrame из результатов
    df = pd.DataFrame(results)
    
    # Сортируем по номеру кадра
    df = df.sort_values('frame').reset_index(drop=True)
    
    # Сохраняем результаты в CSV
    output_file = output_dir / f"{model_name.lower()}_{sequence_id}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Сохранены результаты для {model_name} на {sequence_id} в {output_file}")
    
    return df


def main():
    """Основная функция запуска классификаторов"""
    args = parse_args()
    
    # Преобразования для изображений (стандартный пайплайн для pretrained моделей)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Находим последовательности в директории
    seq_dir = Path(args.seq_dir)
    sequences = find_sequence_ids(seq_dir)
    
    if not sequences:
        print(f"Ошибка: Не найдены последовательности в {seq_dir}")
        return
        
    print(f"Найдены последовательности: {sequences}")
    
    # Загружаем и запускаем каждую модель
    model_names = args.models.split(',')
    
    for model_name in model_names:
        model_name = model_name.strip()
        print(f"\nЗагрузка модели {model_name}...")
        
        # Загружаем модель
        model, feature_layer = load_model(model_name, args.checkpoints_dir, args.device)
        
        if model is None:
            print(f"Пропускаем модель {model_name}")
            continue
            
        # Обрабатываем каждую последовательность
        for seq_id in sequences:
            print(f"Обработка последовательности {seq_id} с моделью {model_name}...")
            
            # Запускаем модель на последовательности
            run_classifier_on_sequence(
                model, model_name, feature_layer, seq_dir, seq_id,
                transform, args.batch_size, args.device, args.out_dir
            )


if __name__ == "__main__":
    main() 