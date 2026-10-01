import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc


def generate_hypercube(n_vars, n_points, bounds, seed=42):
    """
    Генерирует план эксперимента по методу латинского гиперкуба (LHS).

    Параметры:
    n_vars - количество переменных (2 или 3)
    n_points - количество экспериментов (30-50)
    bounds - список кортежей (min, max) для каждой переменной
    seed - для воспроизводимости

    Возвращает:
    numpy array размером (n_points, n_vars)
    """
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
    lhs_samples = sampler.random(n=n_points)

    # Масштабируем в реальные диапазоны
    scaled_samples = []
    for i in range(n_vars):
        low, high = bounds[i]
        scaled = low + lhs_samples[:, i] * (high - low)
        scaled_samples.append(scaled)

    return np.array(scaled_samples).T


def get_default_bounds(var_names):
    """
    Возвращает диапазоны для выбранных XMV.
        'ReactorTempSP (ramp_reactor_temp): (119.0 – 127.0 °C),
        'ReactorPressSP (ramp_reactor_pressure)': (2700 – 2850 кПа),
        'ReactorLevelSP (ramp_reactor_level)': (55 – 75%),
        'ProductionSP (ramp_production)': (18.0 – 24.0),
        'StripLevelSP (ramp_stripper_level)': (35 – 65%),
        'SepLevelSP (ramp_separator_level)': (35 – 65%),
        'SteamValvePosSP (ramp_steam_valve_pos)': (35 – 55%),
        'RecycleValvePosSP (ramp_recycle_valve_pos)': (18 – 26%),
        'AgitatorSpeedSP (ramp_agitator_speed)': (4170 – 230 об/мин),
        'MolePctGSP (ramp_g_in_product)': (0.03 – 0.07),
        'YASP (ramp_ya)': (0.4 – 0.6),
        'YACSP (ramp_yac)': (0.4 – 0.6),
    """
    # Типичные значения XMV (из модели TEP)
    default_values = {
        'ReactorTempSP': 120.40,
        'ReactorPressSP': 2800,
        'ReactorLevelSP': 75.000,
        'ProductionSP': 22.89,
        'StripLevelSP': 50,
        'SepLevelSP': 50,
        'SteamValvePosSP': 47.5,
        'RecycleValvePosSP': 22.2,
        'AgitatorSpeedSP': 200,
        'MolePctGSP': 0.05,
        'YASP': 0.485,
        'YAСSP': 0.485,
    }

    # ±10% от базового значения
    bounds = []
    for name in var_names:
        base = default_values.get(name, 50.0)
        bounds.append((base * 0.9, base * 1.1))

    return bounds


if __name__ == "__main__":
    # ===== ВЫБЕРИТЕ ПЕРЕМЕННЫЕ =====
    var_names = ['ReactorTempSP', 'ReactorPressSP','ProductionSP']

    # 2. Меняем функцию с диапазонами
    def get_default_bounds(var_names):
        bounds = {
            'ReactorTempSP': (120, 128),
            'ReactorPressSP': (2700.0, 2850),
            'ProductionSP': (20, 23),
        }
        return [bounds[name] for name in var_names]
    # ===== ГЕНЕРИРУЕМ ГИПЕРКУБ =====
    n_vars = len(var_names)
    n_points = 30  # для 3 переменных
    bounds = get_default_bounds(var_names)

    # Генерируем точки
    lhs_array = generate_hypercube(n_vars, n_points, bounds, seed=42)

    # Создаём DataFrame
    df_design = pd.DataFrame(lhs_array, columns=var_names)
    df_design.insert(0, 'experiment_id', range(1, n_points + 1))

    # ===== СОХРАНЯЕМ =====
    save_dir = Path("data/Setpoints_cube/Setpoints_cube")
    save_dir.mkdir(parents=True, exist_ok=True)
    Setpoints_cube_path = save_dir / "hypercube_design.csv"
    df_design.to_csv(Setpoints_cube_path, index=False)

    print(f"✅ План эксперимента сохранён: {Setpoints_cube_path}")
    print(f"   Переменные: {var_names}")
    print(f"   Экспериментов: {n_points}")
    print("\n📊 Первые 5 строк:")
    print(df_design.head())
    print("\n📊 Диапазоны:")
    for i, name in enumerate(var_names):
        low, high = bounds[i]
        print(f"   {name}: {low:.2f} — {high:.2f}")