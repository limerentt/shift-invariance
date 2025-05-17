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
            # Антиалиасинговая версия VGG16
            model_path = Path(ckpt_dir) / "aa_vgg16.pth"
            if model_path.exists():
                # Здесь предполагается, что AA-модель имеет такую же архитектуру как обычная VGG16
                model = models.vgg16(pretrained=False)
                # Заменяем maxpool слои на anti-aliased пулинг
                # Для полноценной реализации нужна дополнительная логика
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Загружена AA-VGG16 из {model_path}")
            else:
                print("Ошибка: Не найдены веса AA-VGG16")
                return None, None
                
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
            model_path = Path(ckpt_dir) / "aa_resnet50.pth"
            if model_path.exists():
                # По аналогии с AA-VGG16
                model = models.resnet50(pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Загружена AA-ResNet50 из {model_path}")
            else:
                print("Ошибка: Не найдены веса AA-ResNet50")
                return None, None
                
            features_layer = "avgpool"
            
        except Exception as e:
            print(f"Ошибка при загрузке AA-ResNet50: {e}")
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
    
    # Разбиваем на батчи
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        # Загружаем и преобразуем изображения
        batch_images = []
        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
                continue
        
        # Если в батче нет изображений, пропускаем его
        if not batch_images:
            continue
            
        # Создаем батч тензоров
        batch_tensor = torch.stack(batch_images)
        
        # Получаем признаки и логиты
        features, logits = extract_features(model, feature_layer, batch_tensor, device)
        
        # Сохраняем для каждого изображения в батче
        for j, img_path in enumerate(batch_files):
            # Получаем индекс кадра из имени файла
            frame_idx = int(img_path.stem.split('_')[-1])
            
            # Вычисляем top-1 класс и уверенность
            probs = F.softmax(torch.from_numpy(logits[j]), dim=0).numpy()
            top_class = int(np.argmax(probs))
            top_confidence = float(probs[top_class])
            
            # Сохраняем результаты
            result = {
                'frame': img_path.name,
                'frame_idx': frame_idx,
                'features': features[j],  # сохраняем признаки для дальнейшего анализа
                'top_class': top_class,
                'confidence': top_confidence,
                'logits': logits[j]
            }
            
            results.append(result)
    
    # Если нет результатов, возвращаем None
    if not results:
        print(f"Ошибка: Не получены результаты для последовательности {sequence_id}")
        return None
    
    # Сортируем по индексу кадра
    results.sort(key=lambda x: x['frame_idx'])
    
    # Вычисляем косинусное сходство с первым кадром
    base_features = results[0]['features']
    
    for i, result in enumerate(results):
        cos_sim = compute_cosine_similarity(result['features'], base_features)
        results[i]['cos_sim'] = cos_sim
        
        # Вычисляем разницу в уверенности
        conf_drift = abs(result['confidence'] - results[0]['confidence'])
        results[i]['conf_drift'] = conf_drift
    
    # Создаем DataFrame с основными метриками (без признаков и логитов)
    df_results = pd.DataFrame([
        {
            'frame': r['frame'],
            'frame_idx': r['frame_idx'],
            'top_class': r['top_class'],
            'confidence': r['confidence'],
            'cos_sim': r['cos_sim'],
            'conf_drift': r['conf_drift']
        }
        for r in results
    ])
    
    # Сохраняем результаты в CSV
    csv_path = output_dir / f"{model_name}_{sequence_id}.csv"
    df_results.to_csv(csv_path, index=False)
    
    print(f"Сохранены результаты для {model_name} на {sequence_id} в {csv_path}")
    
    return df_results


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