import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import joblib
import numpy as np
from src.run_cube_4 import run_experiment
from src.main_function_1 import calc_tep_profit

# ============================================================
# 1. ЗАГРУЖАЕМ СОХРАНЁННУЮ МОДЕЛЬ
# ============================================================

data = joblib.load("data/doe_final/model_profit.pkl")
x_opt = data['x_opt']
profit_pred = data['profit_opt']

print("="*70)
print("ПРОВЕРКА ОПТИМУМА ПРИБЫЛИ НА СИМУЛЯТОРЕ")
print("="*70)
print(f"\n🏆 Оптимум из модели:")
print(f"   ReactorTempSP  = {x_opt[0]:.3f} °C")
print(f"   ReactorPressSP = {x_opt[1]:.3f} кПа")
print(f"   ProductionSP   = {x_opt[2]:.3f}")
print(f"   Предсказанная прибыль = {profit_pred:.2f} $/ч")

# ============================================================
# 2. ЗАПУСКАЕМ СИМУЛЯТОР
# ============================================================

sp_values = {
    'ReactorTempSP': x_opt[0],
    'ReactorPressSP': x_opt[1],
    'ProductionSP': x_opt[2]
}

try:
    print(f"\n▶️ Запуск симулятора...")
    result = run_experiment(sp_values, duration=30, steady_threshold=0.001)

    # Считаем прибыль по фактическим данным
    steady_xmeas = result['steady_xmeas']
    steady_xmv = result['steady_xmv']
    steady_data = {**steady_xmeas, **steady_xmv}

    profit_actual, comp = calc_tep_profit(steady_data)

    print(f"\n📊 РЕЗУЛЬТАТ:")
    print(f"   Предсказанная прибыль: {profit_pred:.2f} $/ч")
    print(f"   Фактическая прибыль:   {profit_actual:.2f} $/ч")
    print(f"   Ошибка: {abs(profit_actual - profit_pred):.2f} $/ч")
    print(f"   Относительная ошибка: {abs(profit_actual - profit_pred)/profit_actual*100:.2f}%")

    print(f"\n📊 Компоненты:")
    print(f"   Доход:      {comp['revenue']:.2f} $/ч")
    print(f"   Затраты:    {comp['total_cost']:.2f} $/ч")

    # Сравнение с лучшим DOE
    best_doe_profit = 7234.01
    print(f"\n📊 Сравнение с лучшим DOE:")
    print(f"   Лучший DOE:   {best_doe_profit:.2f} $/ч")
    print(f"   Оптимум:      {profit_actual:.2f} $/ч")
    print(f"   Улучшение:    {profit_actual - best_doe_profit:.2f} $/ч")

    if abs(profit_actual - profit_pred) / profit_actual < 0.02:
        print("\n✅ Модель адекватна (ошибка < 2%)")
    else:
        print("\n⚠️ Модель требует уточнения (ошибка > 2%)")

except Exception as e:
    print(f"\n❌ Эксперимент упал: {e}")
    import traceback
    traceback.print_exc()