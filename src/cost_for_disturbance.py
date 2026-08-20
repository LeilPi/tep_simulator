import sys
from pathlib import Path

# Добавляем корневую папку в sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd
from src.main_function import calc_cost_from_files, calc_tep_cost

# Пути к данным с возмущением
disturbance_xmeas = "data/disturbance/idv_4/xmeas/disturbance_4_xmeas.csv"
disturbance_xmv = "data/disturbance/idv_4/xmv/disturbance_4_xmv.csv"

print("=" * 60)
print("Расчёт целевой функции для сценария с возмущением")
print("=" * 60)

# 1. Рассчитываем стоимость
total_cost, components, df = calc_cost_from_files(disturbance_xmeas, disturbance_xmv)

print(f"\n💰 Итоговая стоимость: {total_cost:.4f}")
print("\n📊 Компоненты стоимости:")
for key, value in components.items():
    print(f"   {key}: {value:.4f}")

# 2. Информация о данных
print(f"\n📈 Данные: {len(df)} строк, {len(df.columns)} колонок")
print(f"   Временной диапазон: {df['time'].min():.2f} - {df['time'].max():.2f} ч")

# 3. Стоимость по времени (для визуализации)
df['cost'] = df.apply(lambda row: calc_tep_cost(row)[0], axis=1)
print(f"\n   Средняя стоимость за всё время: {df['cost'].mean():.4f}")
print(f"   Стандартное отклонение: {df['cost'].std():.4f}")

# 4. Сравнение с базовым сценарием (если есть файлы)
baseline_xmeas = "data/baseline/xmeas/baseline_xmeas.csv"
baseline_xmv = "data/baseline/xmv/baseline_xmv.csv"

try:
    total_cost_base, _, _ = calc_cost_from_files(baseline_xmeas, baseline_xmv)
    print(f"\n📊 Сравнение:")
    print(f"   Базовый сценарий: {total_cost_base:.4f}")
    print(f"   С возмущением: {total_cost:.4f}")
    print(f"   Разница: {total_cost - total_cost_base:.4f}")
    print(f"   Изменение: {(total_cost - total_cost_base) / total_cost_base * 100:.2f}%")
except FileNotFoundError:
    print("\n⚠️ Файлы базового сценария не найдены, сравнение пропущено.")