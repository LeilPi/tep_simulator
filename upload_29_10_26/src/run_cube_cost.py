# src/run_cube_cost.py
"""
Сбор данных по плану data/doe_cost/design/hypercube_cost.csv.
Целевая величина — эксплуатационные затраты по Downs & Vogel (Table 9),
усреднение по участку 20–30 ч.

Уже выполненные эксперименты пропускаются — после сбоя можно просто
запустить скрипт снова, он продолжит с того места, где остановился.

Запуск из корня проекта:
    python src/run_cube_cost.py              # весь план
    python src/run_cube_cost.py --max 2      # пробный запуск на 2 точках
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import run_cost_experiment, save_histories   # noqa: E402

DESIGN = ROOT / 'data/doe_cost/design/hypercube_cost.csv'
OUT_DIR = ROOT / 'data/doe_cost/runs'
RESULTS = OUT_DIR / 'results_cost.csv'
SP_COLS = ['ReactorTempSP', 'ReactorPressSP', 'ProductionSP',
           'ReactorLevelSP', 'AgitatorSpeedSP', 'SteamValvePosSP']


def main(max_experiments=None):
    design = pd.read_csv(DESIGN)
    if max_experiments:
        design = design.head(max_experiments)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    done_ids = set(done['experiment_id']) if len(done) else set()
    rows = done.to_dict('records') if len(done) else []

    for _, r in design.iterrows():
        exp_id = int(r['experiment_id'])
        if exp_id in done_ids:
            print(f"Эксперимент {exp_id}: уже есть, пропуск")
            continue

        sp = {c: float(r[c]) for c in SP_COLS}
        print(f"\n{'─' * 60}\nЭксперимент {exp_id}/{len(design)}  ({r['point_type']})")
        for k, v in sp.items():
            print(f"   {k:16s} = {v:9.3f}")

        t0 = time.time()
        try:
            res = run_cost_experiment(sp)
        except Exception:
            traceback.print_exc()
            res = None
        if res is None:
            print("   Не удалось, пропуск")
            continue

        save_histories(res, OUT_DIR / f"exp_{exp_id:03d}")
        row = {'experiment_id': exp_id, 'point_type': r['point_type'],
               **res['row'], 'elapsed_s': time.time() - t0}
        rows.append(row)
        pd.DataFrame(rows).sort_values('experiment_id').to_csv(RESULTS, index=False)

        flag = '  ВНИМАНИЕ: давление выше 2895 кПа' if row['p_over_limit'] else ''
        print(f"   затраты = {row['operating_cost']:.2f} $/ч "
              f"(разброс блоков σ = {row['cost_block_std']:.2f}){flag}")

    if rows:
        df = pd.DataFrame(rows)
        print(f"\nГотово: {len(df)} экспериментов, результаты в {RESULTS}")
        print(df['operating_cost'].describe().round(2).to_string())
        print("\nЛучшие 5 (минимум затрат):")
        print(df.nsmallest(5, 'operating_cost')[['experiment_id'] + SP_COLS + ['operating_cost']]
              .round(2).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=None)
    main(ap.parse_args().max)
