import sys
from pathlib import Path
import numpy as np
# Добавляем корневую папку, что бы импортировать директории
main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

import pandas as pd
from pytep.siminterface import SimInterface
from src.simulator_config import DEFAULT_DURATION, DEFAULT_ONSET_TIME, DEFAULT_DISTURBANCE_ID
# Конструкция для преобразования xmv_data в плоский словарь, ибо не смог найти как pytep возвращает его значения
def flatten_xmv(xmv_data):
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

    print(f"Неизвестный тип xmv: {type(xmv_data)}")
    return {}
def run_disturbance_separate(disturbance_id = DEFAULT_DISTURBANCE_ID, onset_time = DEFAULT_ONSET_TIME, duration = DEFAULT_DURATION, base_folder = "data/disturbance", step = 0.05, disturbance_v = 0.1):
    print("Запуск симулятора...")
    tep = SimInterface()
    tep.setup()

    #Инициализируем списки данных
    time_points = []
    xmeas_history = []
    xmv_history = []

    #Определяем названия xmv и xmeas
    xmeas_names = tep.process_data_labels()  # ← сохраняем в переменную
    xmv_names = tep.manipulated_variables  # ← список xmv
    print(f"Найдено xmv: {len(xmv_names)} переменных")
    print(f"Найдено xmeas: {len(xmeas_names)} переменных")

    steps = int(duration/step)
    print(f"Запуск симуляции на {duration} ч, шаг {step} ч, всего {steps} шагов")
    print(f"Возмущение IDV({disturbance_id}) включится в {onset_time} ч")
    disturbance_activated = False
    for i in range(steps):
        try:
            current_time = tep.current_sim_time()
        except TypeError:
            current_time = tep.current_sim_time
            # Проверяем, нужно ли включить возмущение на этом шаге
        if not disturbance_activated and current_time >= onset_time:
            # Изменяем IDV
            current_val = tep.get_idv(disturbance_id)
            new_val = current_val + disturbance_v
            print(f" Изменяем IDV({disturbance_id}) с {current_val:.3f} на {new_val:.3f}")
            tep.set_idv(disturbance_id, new_val)
            disturbance_activated = True
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
         # Выводим ход выполнения
        if (i + 1) % 10 == 0:
            print(f"Шаг {i + 1}/{steps}, время: {sim_time:.2f} ч")

    # Преобразуем в DataFrame
    df_xmeas = pd.DataFrame(xmeas_history)
    df_xmv = pd.DataFrame(xmv_history)
    # Добавляем время (если отсутствует)
    if 'time' not in df_xmeas.columns:
        df_xmeas.insert(0, 'time', time_points)
    if 'time' not in df_xmv.columns:
        df_xmv.insert(0, 'time', time_points)

    # Добавляем disturbance_id в обе таблицы
    df_xmeas['disturbance_id'] = 0
    df_xmv['disturbance_id'] = 0

    # Маркируем данные после включения возмущения
    df_xmeas.loc[df_xmeas['time'] >= onset_time, 'disturbance_id'] = disturbance_id
    df_xmv.loc[df_xmv['time'] >= onset_time, 'disturbance_id'] = disturbance_id

    # Создаём папки с учётом ID возмущения
    disturbance_folder = Path(base_folder) / f"idv_{disturbance_id}"

    # Сохраняем xmeas
    xmeas_folder = disturbance_folder / "xmeas"
    xmeas_folder.mkdir(parents=True, exist_ok=True)
    xmeas_path = xmeas_folder / f"disturbance_{disturbance_id}_xmeas.csv"
    df_xmeas.to_csv(xmeas_path, index=False)
    print(f" Xmeas сохранены в {xmeas_path} (строк: {len(df_xmeas)})")

    # Сохраняем xmv
    xmv_folder = disturbance_folder / "xmv"
    xmv_folder.mkdir(parents=True, exist_ok=True)
    xmv_path = xmv_folder / f"disturbance_{disturbance_id}_xmv.csv"
    df_xmv.to_csv(xmv_path, index=False)
    print(f" Xmv сохранены в {xmv_path} (строк: {len(df_xmv)})")

    # Сбрасываем симуляцию
    tep.reset()

    return df_xmeas, df_xmv

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Запуск симуляции TEP с возмущением")
    parser.add_argument("--disturbance_id", type=int, default=DEFAULT_DISTURBANCE_ID,
                        help="Номер возмущения (1-12)")
    parser.add_argument("--onset", type=float, default=DEFAULT_ONSET_TIME,
                        help="Время включения возмущения (часы)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Общая длительность симуляции")
    parser.add_argument("--value", type=float, default=0.1,
                        help="Величина изменения IDV")
    args = parser.parse_args()

    df_xmeas, df_xmv = run_disturbance_separate(
        disturbance_id=args.disturbance_id,
        onset_time=args.onset,
        duration=args.duration,
        disturbance_v=args.value
    )

    # Выводим информацию о колонках
    print("\n Xmeas колонки:", df_xmeas.columns.tolist())
    print("Xmv колонки:", df_xmv.columns.tolist())

    # Выводим статистику по disturbance_id
    print(f"\n Статистика disturbance_id в xmeas:")
    print(df_xmeas['disturbance_id'].value_counts().sort_index())




