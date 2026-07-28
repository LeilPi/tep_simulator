import sys
from pathlib import Path

import argparse
from pytep.siminterface import SimInterface
from src.simulator_config import DEFAULT_DURATION, DEFAULT_ONSET_TIME, DEFAULT_DISTURBANCE_ID
from src.save_data import save_simulation_data


def run_disturbance(disturbance_id, onset_time, duration=DEFAULT_DURATION, save_path=None, disturbance_value=0.1):

    if save_path is None:
        save_path = f"data/disturbance/idv_{disturbance_id}_run_001.csv"

    tep = SimInterface()
    tep.setup()

    # Часть 1: до возмущения
    if onset_time > 0:
        print(f"Запуск симуляции до возмущения: {onset_time} часов...")
        tep.simulate(duration=onset_time)
    else:
        print("Возмущение включается сразу.")

    # Изменяем IDV
    current_val = tep.get_idv(disturbance_id)
    new_val = current_val + disturbance_value
    print(f"Изменяем IDV({disturbance_id}) с {current_val:.3f} на {new_val:.3f}")
    tep.set_idv(disturbance_id, new_val)

    # Часть 2: после возмущения
    remaining = duration - onset_time
    if remaining > 0:
        print(f"Продолжение симуляции с возмущением на {remaining} часов...")
        tep.simulate(duration=remaining)
    else:
        print("Оставшееся время равно нулю.")

    # 5. Получаем данные (создаём КОПИЮ, чтобы не менять оригинал)
    df = tep.process_data.copy()
    print(f"Получено {len(df)} строк данных.")

    # 6. Добавляем disturbance_id в копию
    if 'time' in df.columns:
        df['disturbance_id'] = 0
        df.loc[df['time'] >= onset_time, 'disturbance_id'] = disturbance_id
    else:
        print("Колонка 'time' не найдена, disturbance_id будет установлен как константа.")
        df['disturbance_id'] = disturbance_id

    # 7. Сохраняем
    save_simulation_data(df, save_path, disturbance_id=disturbance_id)

    # 8. Сбрасываем (оригинал `tep.process_data` не тронут, ошибки не будет)
    tep.reset()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disturbance_id", type=int, default=DEFAULT_DISTURBANCE_ID)
    parser.add_argument("--onset", type=float, default=DEFAULT_ONSET_TIME)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--value", type=float, default=0.1)
    args = parser.parse_args()

    run_disturbance(args.disturbance_id, args.onset, args.duration, args.save, args.value)