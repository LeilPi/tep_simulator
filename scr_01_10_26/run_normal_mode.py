# src/run_nominal_extended.py

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd
import numpy as np
from src.run_cube_5 import run_experiment_with_retry
from src.main_function_1 import calc_tep_profit

# ============================================================
# НОМИНАЛЬНЫЙ РЕЖИМ TEP (Mode 1, base case)
# ============================================================
# Значения из Downs & Vogel (1993), Table 3, 4

NOMINAL_SP = {
    'ReactorTempSP':   122.9,    # °C
    'ReactorPressSP':  2800.0,   # кПа
    'ProductionSP':    22.89,    # отн. ед.
    'ReactorLevelSP':  75.0,     # %  (Table 4: Reactor level = 75.000%)
    'AgitatorSpeedSP': 200.0,    # об/мин (XMV(12)=50% → 200 об/мин)
    'SteamValvePosSP': 47.446,   # %  (XMV(9) = 47.446)
}

print("=" * 70)
print("ЗАПУСК В НОМИНАЛЬНОМ РЕЖИМЕ (Mode 1, base case)")
print("=" * 70)
print(f"\n📋 Уставки номинального режима:")
for k, v in NOMINAL_SP.items():
    print(f"   {k:20s} = {v}")

# ============================================================
# ЗАПУСК СИМУЛЯЦИИ
# ============================================================

print(f"\n▶️ Запуск симулятора на 30 часов...")

result = run_experiment_with_retry(
    sp_values=NOMINAL_SP,
    duration=30,
    steady_threshold=0.001,
    max_retries=3
)

if result is None:
    print("\n❌ Эксперимент не удался")
    sys.exit(1)

# ============================================================
# РЕЗУЛЬТАТЫ
# ============================================================

profit = result['profit']
components = result['components']

print(f"\n{'=' * 70}")
print("РЕЗУЛЬТАТЫ НОМИНАЛЬНОГО РЕЖИМА")
print(f"{'=' * 70}")

print(f"\n📊 Экономика (факт из симулятора):")
print(f"   Доход:              {components['revenue']:10.2f} $/ч")
print(f"   Потери с продувкой: {components['purge_loss']:10.2f} $/ч")
print(f"   Потери с продуктом: {components['product_loss']:10.2f} $/ч")
print(f"   Компрессор:         {components['compressor_cost']:10.2f} $/ч")
print(f"   Пар:                {components['steam_cost']:10.2f} $/ч")
print(f"   ────────────────────────────────")
print(f"   Затраты (итого):    {components['total_cost']:10.2f} $/ч")
print(f"   ПРИБЫЛЬ:            {profit:10.2f} $/ч")

print(f"\n📊 Стационарность:")
print(f"   is_steady = {result['is_steady']}")
print(f"   Начало стационара: t = {result['steady_start_time']:.2f} ч")

# ============================================================
# СРАВНЕНИЕ С ОПТИМУМОМ
# ============================================================

OPTIMUM_FACT = 7585.44   # из проверки расширенного оптимума

print(f"\n{'=' * 70}")
print("СРАВНЕНИЕ")
print(f"{'=' * 70}")

print(f"\n📊 Прибыль в разных режимах:")
print(f"   Номинальный (Mode 1): {profit:10.2f} $/ч")
print(f"   Оптимум RTO (факт):   {OPTIMUM_FACT:10.2f} $/ч")

delta = OPTIMUM_FACT - profit
print(f"\n📈 Прирост от RTO:")
print(f"   Δ = {delta:+.2f} $/ч")
print(f"   Δ % = {100 * delta / profit:+.2f}%")

# ============================================================
# СОХРАНЕНИЕ
# ============================================================

output = {
    'mode': 'nominal',
    **NOMINAL_SP,
    'revenue': components['revenue'],
    'purge_loss': components['purge_loss'],
    'product_loss': components['product_loss'],
    'compressor_cost': components['compressor_cost'],
    'steam_cost': components['steam_cost'],
    'total_cost': components['total_cost'],
    'profit': profit,
    'is_steady': result['is_steady'],
}

save_path = "data/doe_extended/nominal_mode.csv"
pd.DataFrame([output]).to_csv(save_path, index=False)
print(f"\n💾 Сохранено в {save_path}")

# ============================================================
# ИТОГОВАЯ ТАБЛИЦА
# ============================================================

print(f"\n{'=' * 70}")
print("ИТОГОВАЯ ТАБЛИЦА ДЛЯ ДИПЛОМА")
print(f"{'=' * 70}")
print(f"""
┌──────────────────────────┬──────────────┬──────────────┐
│ Режим                    │ Прибыль $/ч  │ Δ к номиналу │
├──────────────────────────┼──────────────┼──────────────┤
│ Номинал (Mode 1)         │ {profit:10.2f}   │      —       │
│ Оптимум RTO (симулятор)  │ {OPTIMUM_FACT:10.2f}   │ {OPTIMUM_FACT - profit:+10.2f}   │
└──────────────────────────┴──────────────┴──────────────┘
""")