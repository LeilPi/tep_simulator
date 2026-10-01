import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import gc
import traceback
import subprocess
import os

main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

from pytep.siminterface import SimInterface
from src.main_function_1 import calc_tep_profit


# ============================================================
# 1. ПРИНУДИТЕЛЬНОЕ ЗАВЕРШЕНИЕ MATLAB
# ============================================================

def kill_matlab_forcefully():
    """Принудительно завершает процессы MATLAB."""
    try:
        import matlab.engine
        try:
            matlab.engine.quit_engine()
        except:
            pass
    except:
        pass

    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'MATLAB.exe'],
                           capture_output=True, timeout=5)
            subprocess.run(['taskkill', '/F', '/IM', 'MATLAB_Engine.exe'],
                           capture_output=True, timeout=5)
        except Exception as e:
            print(f"⚠️ Ошибка при завершении MATLAB: {e}")
    time.sleep(1)


# ============================================================
# 2. УСТАНОВКА 6 УСТАВОК
# ============================================================

def set_setpoints(tep, sp_values):
    """Устанавливает 6 уставок через ramp_* методы."""
    mapping = {
        'ReactorTempSP':    ('ramp_reactor_temp',     'температура реактора', '°C'),
        'ReactorPressSP':   ('ramp_reactor_pressure', 'давление реактора',    'кПа'),
        'ProductionSP':     ('ramp_production',       'производство',         'отн.ед.'),
        'ReactorLevelSP':   ('ramp_reactor_level',    'уровень реактора',     '%'),
        'AgitatorSpeedSP':  ('ramp_agitator_speed',   'скорость мешалки',     'об/мин'),
        'SteamValvePosSP':  ('ramp_steam_valve_pos',  'клапан пара',          '%'),
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
# 3. СТАЦИОНАРНОСТЬ
# ============================================================

def find_steady_state_window(data_series, max_window=50, threshold=0.001, min_window=10):
    """Находит окно стационарности по относительной производной."""
    n = len(data_series)
    for window_size in range(min_window, min(max_window, n) + 1):
        recent = data_series[-window_size:]
        diffs = np.abs(np.diff(recent))
        avg_diff = np.mean(diffs) if len(diffs) > 0 else 1.0
        mean_val = np.mean(recent)
        rel_deriv = avg_diff / mean_val if mean_val != 0 else avg_diff
        if rel_deriv < threshold:
            return n - window_size, True
    return max(0, n - min_window), False


# ============================================================
# 4. ОДИН ЭКСПЕРИМЕНТ
# ============================================================

def run_experiment(sp_values, duration=30, step=0.05, steady_threshold=0.001):
    """Запускает один эксперимент с 6 уставками."""
    print("   Запуск симулятора...")
    tep = SimInterface()
    tep.setup()

    # Установка уставок
    set_setpoints(tep, sp_values)

    time_points = []
    xmeas_history = []
    xmv_history = []

    steps = int(duration / step)
    print(f"      Симуляция {duration} ч, шаг {step} ч, {steps} шагов")

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
                print(f"      ⚠️ Зависание на {sim_time:.2f} ч. Прерывание.")
                tep.reset()
                gc.collect()
                return None

    # DataFrame
    df_xmeas = pd.DataFrame(xmeas_history)
    df_xmv = pd.DataFrame(xmv_history)

    if 'time' not in df_xmeas.columns:
        df_xmeas.insert(0, 'time', time_points)
    if 'time' not in df_xmv.columns:
        df_xmv.insert(0, 'time', time_points)

    # Стационарность
    key_var = 'Reactor Temperature'
    if key_var not in df_xmeas.columns:
        for col in df_xmeas.columns:
            if col != 'time' and np.issubdtype(df_xmeas[col].dtype, np.number):
                key_var = col
                break

    data_series = df_xmeas[key_var].values
    start_idx, is_steady = find_steady_state_window(
        data_series, max_window=50, threshold=steady_threshold, min_window=10
    )

    # Усреднение стационарного участка
    steady_xmeas_dict = df_xmeas.iloc[start_idx:].mean(numeric_only=True).to_dict()
    steady_xmv_dict = df_xmv.iloc[start_idx:].mean(numeric_only=True).to_dict()

    # Прибыль
    steady_data = {**steady_xmeas_dict, **steady_xmv_dict}
    profit, components = calc_tep_profit(steady_data)

    if isinstance(profit, (pd.Series, pd.DataFrame)):
        profit = float(profit.iloc[0]) if len(profit) > 0 else 0.0
    else:
        profit = float(profit)

    print(f"      ✅ profit = {profit:.2f} $/ч")

    tep.reset()
    gc.collect()

    return {
        'profit': profit,
        'components': components,
        'steady_xmeas': steady_xmeas_dict,
        'steady_xmv': steady_xmv_dict,
        'df_xmeas': df_xmeas,
        'df_xmv': df_xmv,
        'is_steady': is_steady,
        'steady_start_time': time_points[start_idx] if start_idx < len(time_points) else np.nan,
    }


def run_experiment_with_retry(sp_values, max_retries=3, **kwargs):
    """Запуск с повторными попытками."""
    for attempt in range(max_retries):
        try:
            kill_matlab_forcefully()
            result = run_experiment(sp_values, **kwargs)
            if result is None:
                raise RuntimeError("Эксперимент прерван (зависание)")
            kill_matlab_forcefully()
            return result
        except Exception as e:
            print(f"      ⚠️ Попытка {attempt + 1}/{max_retries}: {e}")
            kill_matlab_forcefully()
            if attempt == max_retries - 1:
                return None
            time.sleep(3)
            gc.collect()


# ============================================================
# 5. ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ
# ============================================================

def collect_doe_data(design_path, output_dir="data/doe_extended/runs",
                     max_experiments=None, duration=30, steady_threshold=0.001):
    """Запускает все эксперименты из расширенного плана."""
    print("\n" + "=" * 70)
    print("РАСШИРЕННЫЙ СБОР ДАННЫХ (6 УСТАВОК)")
    print("=" * 70)

    df_design = pd.read_csv(design_path)
    print(f"📋 План: {len(df_design)} экспериментов")
    print(f"   Колонки: {df_design.columns.tolist()}")

    if max_experiments:
        df_design = df_design.head(max_experiments)
        print(f"   Тестовый режим: {max_experiments}")

    sp_cols = [c for c in df_design.columns if c != 'experiment_id']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    steady_records = []

    for idx, row in df_design.iterrows():
        exp_id = int(row['experiment_id'])
        print(f"\n{'─' * 60}")
        print(f"📊 Эксперимент {exp_id}/{len(df_design)}")
        print(f"{'─' * 60}")

        sp_values = {c: row[c] for c in sp_cols}
        for k, v in sp_values.items():
            print(f"   {k} = {v:.3f}")

        start_time = time.time()
        try:
            result = run_experiment_with_retry(
                sp_values=sp_values,
                duration=duration,
                steady_threshold=steady_threshold,
                max_retries=3
            )

            if result is None:
                print(f"   ⏭️ Пропущен")
                continue

            elapsed = time.time() - start_time
            profit = result['profit']

            # Сохраняем
            exp_dir = output_dir / f"exp_{exp_id:03d}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            result['df_xmeas'].to_csv(exp_dir / "xmeas_history.csv", index=False)
            result['df_xmv'].to_csv(exp_dir / "xmv_history.csv", index=False)
            pd.DataFrame([result['steady_xmeas']]).to_csv(exp_dir / "steady_xmeas.csv", index=False)
            pd.DataFrame([result['steady_xmv']]).to_csv(exp_dir / "steady_xmv.csv", index=False)

            # Результат
            result_row = {
                'experiment_id': exp_id,
                'profit': profit,
                'is_steady': result['is_steady'],
                'elapsed_time': elapsed,
                **sp_values
            }
            all_results.append(result_row)

            steady_row = {
                'experiment_id': exp_id,
                'profit': profit,
                **{f'xmeas_{k}': v for k, v in result['steady_xmeas'].items()},
                **{f'xmv_{k}': v for k, v in result['steady_xmv'].items()}
            }
            steady_records.append(steady_row)

            print(f"   ✅ profit = {profit:.2f} $/ч (время {elapsed:.1f} с)")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            traceback.print_exc()

        # Промежуточное сохранение
        if all_results:
            pd.DataFrame(all_results).to_csv(output_dir / "results_extended.csv", index=False)
        if steady_records:
            pd.DataFrame(steady_records).to_csv(output_dir / "steady_data_extended.csv", index=False)

    # Финал
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(output_dir / "results_extended.csv", index=False)
        print(f"\n{'=' * 70}")
        print(f"✅ Готово! Экспериментов: {len(df_res)}")
        print(f"   Min profit: {df_res['profit'].min():.2f}")
        print(f"   Max profit: {df_res['profit'].max():.2f}")
        print(f"   Mean:       {df_res['profit'].mean():.2f}")

        # Топ-5
        print(f"\n🏆 Топ-5:")
        top5 = df_res.nlargest(5, 'profit')[['experiment_id'] + sp_cols + ['profit']]
        print(top5.to_string(index=False))

        return df_res
    else:
        print("\n❌ Нет данных")
        return None


# ============================================================
# 6. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    design_path = "data/doe_extended/design/hypercube_extended_v2.csv"

    collect_doe_data(
        design_path=design_path,
        output_dir="data/doe_extended/runs",
        max_experiments=None,   # ← для теста, потом поставить None
        duration=30,
        steady_threshold=0.001
    )