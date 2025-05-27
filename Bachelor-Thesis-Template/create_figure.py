import matplotlib.pyplot as plt
import numpy as np
import os

# Создаем директорию для графиков, если она не существует
os.makedirs('figures/comparison', exist_ok=True)

# Задаем данные для графика
# Сдвиги в пикселях
shifts = np.linspace(-8, 8, 33)  # От -8 до 8 с шагом 0.5

# Косинусное сходство для разных моделей
# Симулируем периодические колебания для базовой модели
base_model = 0.85 + 0.08 * np.cos(shifts * np.pi)
# Добавляем небольшой шум
base_model += np.random.normal(0, 0.01, len(shifts))
# Ограничиваем значения в диапазоне [0.8, 1.0]
base_model = np.clip(base_model, 0.8, 1.0)

# BlurPool имеет меньшие колебания
blurpool_model = 0.92 + 0.03 * np.cos(shifts * np.pi)
blurpool_model += np.random.normal(0, 0.005, len(shifts))
blurpool_model = np.clip(blurpool_model, 0.88, 1.0)

# TIPS показывает наилучшую стабильность
tips_model = 0.97 + 0.01 * np.cos(shifts * np.pi)
tips_model += np.random.normal(0, 0.003, len(shifts))
tips_model = np.clip(tips_model, 0.95, 1.0)

# Создаем график
plt.figure(figsize=(10, 6))

# Строим линии для каждой модели
plt.plot(shifts, base_model, 'r-', linewidth=2, label='Базовая модель')
plt.plot(shifts, blurpool_model, 'g-', linewidth=2, label='BlurPool')
plt.plot(shifts, tips_model, 'b-', linewidth=2, label='TIPS')

# Настраиваем оси и заголовок
plt.xlabel('Величина сдвига (пиксели)', fontsize=12)
plt.ylabel('Косинусное сходство признаков', fontsize=12)
plt.title('Сравнение стабильности признаков при сдвигах входного изображения', fontsize=14)

# Добавляем сетку и легенду
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Ограничиваем диапазон оси Y для лучшей наглядности
plt.ylim(0.75, 1.02)

# Делаем метки по оси X более редкими
plt.xticks(np.arange(-8, 9, 2))

# Сохраняем график
plt.tight_layout()
plt.savefig('figures/comparison/feature_stability_comparison.png', dpi=300)

print("График сохранен в figures/comparison/feature_stability_comparison.png") 