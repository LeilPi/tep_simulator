# src/validate_extended_optimum.py

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import joblib
import numpy as np
from src.run_cube_5 import run_experiment_with_retry
from src.main_function_1 import calc_tep_profit

# ============================================================
# 1. ЗАГРУЗКА МОДЕЛИ
# ============================================================

print("=" * 70)
print("ПРОВЕРКА РАСШИРЕННОГО ОПТИМУМА НА СИМУЛЯТОРЕ")
print("=" * 70)

data = joblib.load("data/doe_extended/model_profit_extended.pkl")
x_opt = data['x_opt']
profit_pred = data['profit_opt']
feature_cols = data['feature_cols']

print(f"\n🏆 Оптимум из модели:")
for i, name in enumerate(feature_cols):
    print(f"   {name:20s} = {x_opt[i]:.3f}")
print(f"   Предсказанная прибыль = {profit_pred:.2f} $/ч")

# ============================================================
# 2. ЗАПУСК СИМУЛЯТОРА
# ============================================================

sp_values = {name: x_opt[i] for i, name in enumerate(feature_cols)}

print(f"\n▶️ Запуск симулятора TEP...")

try:
    result = run_experiment_with_retry(
        sp_values=sp_values,
        duration=30,
        steady_threshold=0.001,
        max_retries=3
    )

    if result is None:
        print("\n❌ Эксперимент не удался (зависание или ошибка)")
    else:
        profit_actual = result['profit']

        print(f"\n{'=' * 70}")
        print("РЕЗУЛЬТАТ ПРОВЕРКИ")
        print(f"{'=' * 70}")

        print(f"\n📊 Прибыль:")
        print(f"   Предсказание модели: {profit_pred:.2f} $/ч")
        print(f"   Факт (симулятор):    {profit_actual:.2f} $/ч")

        error = abs(profit_actual - profit_pred)
        error_pct = error / profit_actual * 100

        print(f"\n📊 Ошибка модели:")
        print(f"   Абсолютная: {error:.2f} $/ч")
        print(f"   Относительная: {error_pct:.2f}%")

        if error_pct < 2:
            print(f"   ✅ Модель адекватна (ошибка < 2%)")
        elif error_pct < 5:
            print(f"   ⚠️ Модель приемлема (ошибка 2–5%)")
        else:
            print(f"   ❌ Модель требует уточнения (ошибка > 5%)")

        # Сравнение с номиналом
        NOMINAL_PROFIT = 7225.26

        print(f"\n📊 Сравнение с номиналом (Mode 1):")
        print(f"   Номинал:             {NOMINAL_PROFIT:.2f} $/ч")
        print(f"   Оптимум (факт):      {profit_actual:.2f} $/ч")
        print(f"   Прирост:             {profit_actual - NOMINAL_PROFIT:+.2f} $/ч")
        print(f"   Прирост %:           {100 * (profit_actual - NOMINAL_PROFIT) / NOMINAL_PROFIT:+.2f}%")

        # Проверка стационарности
        print(f"\n📊 Качество данных:")
        print(f"   is_steady = {result['is_steady']}")
        print(f"   Начало стационара: t = {result['steady_start_time']:.2f} ч")

        # Сохранение
        import pandas as pd
        output = {
            **sp_values,
            'profit_pred': profit_pred,
            'profit_actual': profit_actual,
            'error': error,
            'error_pct': error_pct,
            'is_steady': result['is_steady'],
        }
        pd.DataFrame([output]).to_csv("data/doe_extended/validation_result.csv", index=False)
        print(f"\n💾 Сохранено в data/doe_extended/validation_result.csv")

except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()