import sys
from pathlib import Path
import numpy as np

# Добавляем корневую папку в sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd
from pytep.siminterface import SimInterface
from src.simulator_config import DEFAULT_DURATION


def flatten_xmv(xmv_data):
    """
    Преобразует xmv_data в плоский словарь, независимо от структуры.
    """
    if isinstance(xmv_data, dict):
        return xmv_data

    if isinstance(xmv_data, pd.Series):
        return xmv_data.to_dict()

    if isinstance(xmv_data, pd.DataFrame):
        return xmv_data.iloc[0].to_dict()

    if isinstance(xmv_data, (list, tuple)):
        if len(xmv_data) == 1 and isinstance(xmv_data[0], (dict, list, np.ndarray)):
            return flatten_xmv(xmv_data[0])
        else:
            return {f"xmv_{i}": val for i, val in enumerate(xmv_data)}

    if isinstance(xmv_data, np.ndarray):
        if xmv_data.ndim > 1:
            return flatten_xmv(xmv_data[0])
        else:
            return {f"xmv_{i}": val for i, val in enumerate(xmv_data)}

    print(f"⚠️ Неизвестный тип xmv: {type(xmv_data)}")
    return {}


def run_baseline_separate(duration=DEFAULT_DURATION,
                          base_folder="data/baseline",
                          step=0.05):
    """
    Запускает базовую симуляцию без возмущений и сохраняет:
    - xmeas (измеряемые переменные) в data/baseline/xmeas/
    - xmv (управляющие переменные) в data/baseline/xmv/
    """
    print("Инициализация симулятора...")
    tep = SimInterface()
    tep.setup()

    # Списки для сбора данных
    time_points = []
    xmeas_history = []
    xmv_history = []

    # Получаем список названий xmv
    xmv_names = tep.manipulated_variables
    print(f"Найдено xmv: {len(xmv_names)} переменных")

    # Получаем список названий xmeas
    try:
        xmeas_names = tep.process_data_labels()
    except TypeError:
        xmeas_names = tep.process_data_labels
    print(f"Найдено xmeas: {len(xmeas_names)} переменных")

    steps = int(duration / step)
    print(f"Запуск симуляции на {duration} ч, шаг {step} ч, всего {steps} шагов")

    for i in range(steps):
        # Запускаем на один интервал
        tep.simulate(duration=step)

        # Получаем текущие xmeas (DataFrame)
        current_data = tep.current_process_data()
        if isinstance(current_data, pd.DataFrame):
            current_xmeas = current_data.iloc[-1].to_dict()
        else:
            current_xmeas = current_data

        # Получаем текущие xmv (преобразуем в плоский словарь)
        raw_xmv = tep.current_manipulated_variables()
        current_xmv = flatten_xmv(raw_xmv)

        # Получаем время
        try:
            sim_time = tep.current_sim_time()
        except TypeError:
            sim_time = tep.current_sim_time

        # Сохраняем
        time_points.append(sim_time)
        xmeas_history.append(current_xmeas)
        xmv_history.append(current_xmv)

        if (i + 1) % 10 == 0:
            print(f"Шаг {i + 1}/{steps}, время: {sim_time:.2f} ч")

    # Преобразуем в DataFrame
    df_xmeas = pd.DataFrame(xmeas_history)
    df_xmv = pd.DataFrame(xmv_history)

    # ВРЕМЯ УЖЕ ЕСТЬ В xmeas! Проверяем и добавляем только при необходимости.
    if 'time' not in df_xmeas.columns:
        df_xmeas.insert(0, 'time', time_points)

    # Для xmv время обычно отсутствует, поэтому добавляем
    if 'time' not in df_xmv.columns:
        df_xmv.insert(0, 'time', time_points)

    # Сохраняем xmeas
    xmeas_folder = Path(base_folder) / "xmeas"
    xmeas_folder.mkdir(parents=True, exist_ok=True)
    xmeas_path = xmeas_folder / "baseline_xmeas.csv"
    df_xmeas.to_csv(xmeas_path, index=False)
    print(f"✅ Xmeas сохранены в {xmeas_path} (строк: {len(df_xmeas)})")

    # Сохраняем xmv
    xmv_folder = Path(base_folder) / "xmv"
    xmv_folder.mkdir(parents=True, exist_ok=True)
    xmv_path = xmv_folder / "baseline_xmv.csv"
    df_xmv.to_csv(xmv_path, index=False)
    print(f"✅ Xmv сохранены в {xmv_path} (строк: {len(df_xmv)})")

    # Сбрасываем симуляцию
    tep.reset()

    return df_xmeas, df_xmv


if __name__ == "__main__":
    df_xmeas, df_xmv = run_baseline_separate()

    # Выводим информацию о колонках
    print("\n📊 Xmeas колонки:", df_xmeas.columns.tolist())
    print("📊 Xmv колонки:", df_xmv.columns.tolist())