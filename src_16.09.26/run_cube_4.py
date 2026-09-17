import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import gc
import traceback

main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

from pytep.siminterface import SimInterface
from src.main_function import calc_tep_cost


# ============================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def find_steady_state_window(data_series, max_window=50, threshold=0.001, min_window=10):
    """
    Находит окно, в котором процесс стационарен.
    Используется относительная производная (средняя разность / среднее значение).
    """
    n = len(data_series)
    for window_size in range(min_window, min(max_window, n) + 1):
        recent = data_series[-window_size:]
        diffs = np.abs(np.diff(recent))
        avg_diff = np.mean(diffs) if len(diffs) > 0 else 1.0
        mean_val = np.mean(recent)
        relative_derivative = avg_diff / mean_val if mean_val != 0 else avg_diff
        if relative_derivative < threshold:
            return n - window_size, True
    return max(0, n - min_window), False


def set_setpoints(tep, sp_values):
    """
    Устанавливает уставки через соответствующие ramp_* методы.
    """
    mapping = {
        'ReactorTempSP': ('ramp_reactor_temp', 'температура реактора', '°C'),
        'ReactorPressSP': ('ramp_reactor_pressure', 'давление реактора', 'кПа'),
        'ProductionSP': ('ramp_production', 'производство', 'отн. ед.'),
    }

    for sp_name, value in sp_values.items():
        if sp_name in mapping:
            method_name, display_name, unit = mapping[sp_name]
            method = getattr(tep, method_name)
            method(target_val=value, duration=0.01)
            print(f"      {display_name} ({sp_name}) = {value:.3f} {unit}")
        else:
            print(f"      ⚠️ Неизвестная уставка: {sp_name}")


# ============================================================
# 2. ЗАПУСК ОДНОГО ЭКСПЕРИМЕНТА
# ============================================================

def run_experiment(sp_values, duration=30, step=0.05, steady_threshold=0.001):
    """Запускает один эксперимент с заданными уставками."""
    print("   Запуск симулятора...")
    tep = SimInterface()
    tep.setup()

    # 1. Устанавливаем уставки
    set_setpoints(tep, sp_values)

    # 2. Сбор данных
    time_points = []
    xmeas_history = []
    xmv_history = []

    steps = int(duration / step)
    print(f"      Симуляция до {duration} ч, шаг {step} ч, всего {steps} шагов")

    for i in range(steps):
        tep.simulate(duration=step)

        current_data = tep.current_process_data()
        if isinstance(current_data, pd.DataFrame):
            current_xmeas = current_data.iloc[-1].to_dict()
        else:
            current_xmeas = current_data

        raw_xmv = tep.current_manipulated_variables()
        if isinstance(raw_xmv, pd.DataFrame):
            current_xmv = raw_xmv.iloc[0].to_dict()
        else:
            current_xmv = raw_xmv

        try:
            sim_time = tep.current_sim_time()
        except TypeError:
            sim_time = tep.current_sim_time

        time_points.append(sim_time)
        xmeas_history.append(current_xmeas)
        xmv_history.append(current_xmv)

        if (i + 1) % 10 == 0:
            print(f"      Шаг {i + 1}/{steps}, время: {sim_time:.2f} ч")

        # Защита от зависания
        if i > 3:
            last_times = time_points[-3:]
            if len(last_times) == 3 and all(t == last_times[0] for t in last_times):
                print(f"      ⚠️ Время застыло на {sim_time:.2f} ч. Прерывание эксперимента.")
                tep.reset()
                gc.collect()
                return None

    # 3. Формируем DataFrame
    df_xmeas = pd.DataFrame(xmeas_history)
    df_xmv = pd.DataFrame(xmv_history)

    if 'time' not in df_xmeas.columns:
        df_xmeas.insert(0, 'time', time_points)
    if 'time' not in df_xmv.columns:
        df_xmv.insert(0, 'time', time_points)

    # 4. Находим стационарный участок по температуре реактора
    key_var = 'Reactor Temperature'
    if key_var not in df_xmeas.columns:
        for col in df_xmeas.columns:
            if col != 'time' and np.issubdtype(df_xmeas[col].dtype, np.number):
                key_var = col
                break
    print(f"      Проверка стационарности по переменной: {key_var}")

    data_series = df_xmeas[key_var].values
    start_idx, is_steady = find_steady_state_window(
        data_series, max_window=50, threshold=steady_threshold, min_window=10
    )

    print(f"      Стационарный участок: индексы {start_idx}-{len(data_series) - 1}")
    print(f"      Стационарность: {'ДА' if is_steady else 'НЕТ (берём последние точки)'}")

    # 5. Усредняем стационарный участок
    steady_xmeas_dict = df_xmeas.iloc[start_idx:].mean(numeric_only=True).to_dict()
    steady_xmv_dict = df_xmv.iloc[start_idx:].mean(numeric_only=True).to_dict()

    # 6. Рассчитываем целевую функцию
    steady_data = {**steady_xmeas_dict, **steady_xmv_dict}
    total_cost, components = calc_tep_cost(steady_data)

    if isinstance(total_cost, (pd.Series, pd.DataFrame)):
        total_cost = float(total_cost.iloc[0]) if len(total_cost) > 0 else 0.0
    else:
        total_cost = float(total_cost)

    print(f"      total_cost = {total_cost:.4f}")

    # 7. Закрываем симуляцию
    tep.reset()
    gc.collect()

    return {
        'total_cost': total_cost,
        'steady_xmeas': steady_xmeas_dict,
        'steady_xmv': steady_xmv_dict,
        'df_xmeas': df_xmeas,
        'df_xmv': df_xmv,
        'is_steady': is_steady,
        'steady_start_time': time_points[start_idx] if start_idx < len(time_points) else np.nan
    }


def run_experiment_with_retry(sp_values, max_retries=2, **kwargs):
    """Запускает эксперимент с повторными попытками при ошибках."""
    for attempt in range(max_retries):
        try:
            result = run_experiment(sp_values, **kwargs)
            if result is None:
                raise RuntimeError("Эксперимент прерван из-за зависания")
            return result
        except Exception as e:
            print(f"      ⚠️ Попытка {attempt + 1}/{max_retries} не удалась: {e}")
            traceback.print_exc()
            if attempt == max_retries - 1:
                raise
            print("      🔄 Перезапуск...")
            time.sleep(2)
            gc.collect()


# ============================================================
# 3. ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ
# ============================================================

def collect_doe_data(design_path, output_dir="data/doe_final",
                     max_experiments=None, duration=30, steady_threshold=0.001):
    """Запускает все эксперименты из плана DOE и собирает данные."""
    print("\n" + "=" * 70)
    print("СБОР ДАННЫХ ДЛЯ МОДЕЛИ RTO (ФИНАЛЬНАЯ ВЕРСИЯ)")
    print("=" * 70)

    # Загружаем план
    df_design = pd.read_csv(design_path)
    print(f"📋 Загружен план: {len(df_design)} экспериментов")
    print(f"   Колонки: {df_design.columns.tolist()}")

    if max_experiments:
        df_design = df_design.head(max_experiments)
        print(f"   Тестовый режим: только {max_experiments} экспериментов")

    # Определяем колонки с уставками
    sp_cols = ['ReactorTempSP', 'ReactorPressSP', 'ProductionSP']
    print(f"   Уставки: {sp_cols}")

    # Папка для сохранения
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Списки для общих результатов
    all_results = []
    steady_records = []

    # Запускаем эксперименты
    for idx, row in df_design.iterrows():
        exp_id = int(row['experiment_id'])
        print(f"\n{'─' * 60}")
        print(f"📊 Эксперимент {exp_id}/{len(df_design)}")
        print(f"{'─' * 60}")

        # Формируем словарь уставок
        sp_values = {col: row[col] for col in sp_cols}
        for key, val in sp_values.items():
            print(f"   {key} = {val:.3f}")

        start_time = time.time()
        try:
            # Запускаем эксперимент
            result = run_experiment_with_retry(
                sp_values=sp_values,
                duration=duration,
                steady_threshold=steady_threshold,
                max_retries=2
            )
            elapsed = time.time() - start_time

            # Извлекаем данные
            total_cost = result['total_cost']
            steady_xmeas = result['steady_xmeas']
            steady_xmv = result['steady_xmv']
            is_steady = result['is_steady']
            steady_start_time = result['steady_start_time']

            # Сохраняем полную историю
            exp_dir = output_dir / f"exp_{exp_id:03d}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            result['df_xmeas'].to_csv(exp_dir / "xmeas_history.csv", index=False)
            result['df_xmv'].to_csv(exp_dir / "xmv_history.csv", index=False)

            # Сохраняем усреднённые значения
            pd.DataFrame([steady_xmeas]).to_csv(exp_dir / "steady_xmeas.csv", index=False)
            pd.DataFrame([steady_xmv]).to_csv(exp_dir / "steady_xmv.csv", index=False)

            # Запись в общие результаты
            result_row = {
                'experiment_id': exp_id,
                'total_cost': total_cost,
                'is_steady': is_steady,
                'steady_start_time': steady_start_time,
                'elapsed_time': elapsed,
                **sp_values
            }
            all_results.append(result_row)

            # Запись усреднённых данных для модели
            steady_row = {
                'experiment_id': exp_id,
                'total_cost': total_cost,
                'is_steady': is_steady,
                **{f'xmeas_{k}': v for k, v in steady_xmeas.items()},
                **{f'xmv_{k}': v for k, v in steady_xmv.items()}
            }
            steady_records.append(steady_row)

            print(f"   ✅ total_cost = {total_cost:.4f} (время: {elapsed:.1f} с)")

        except Exception as e:
            print(f"   ❌ ОШИБКА в эксперименте {exp_id}: {e}")
            traceback.print_exc()

        # Промежуточное сохранение
        if all_results:
            pd.DataFrame(all_results).to_csv(output_dir / "results.csv", index=False)
        if steady_records:
            pd.DataFrame(steady_records).to_csv(output_dir / "steady_state_data.csv", index=False)

    # Финальное сохранение
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_dir / "results.csv", index=False)
        print(f"\n✅ Общие результаты сохранены в {output_dir / 'results.csv'}")
        print(f"   Всего экспериментов: {len(df_results)}")
        print(f"   Минимальная стоимость: {df_results['total_cost'].min():.4f}")
        print(f"   Максимальная стоимость: {df_results['total_cost'].max():.4f}")

    if steady_records:
        df_steady = pd.DataFrame(steady_records)
        df_steady.to_csv(output_dir / "steady_state_data.csv", index=False)
        print(f"✅ Усреднённые данные для модели сохранены в {output_dir / 'steady_state_data.csv'}")
        print(f"   Размер: {df_steady.shape}")

    return df_results if all_results else None


# ============================================================
# 4. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    # Путь к вашему гиперкубу
    design_path = "data/Setpoints_cube/Setpoints_cube/hypercube_design.csv"

    # Запуск сбора данных
    collect_doe_data(
        design_path=design_path,
        output_dir="data/doe_final",
        max_experiments=30,  # Для теста — 2 эксперимента, потом удалить
        duration=30,
        steady_threshold=0.001
    )