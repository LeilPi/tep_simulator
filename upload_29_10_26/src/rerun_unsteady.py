# src/rerun_unsteady.py
"""
Повторный прогон опытов, которые не вышли на стационар к 20 ч.

Критерий «не установился»: разброс затрат по 2-часовым блокам больше
2 × медианы по всему плану (в первом прогоне — опыты 11, 23, 27).
Такие точки прогоняются дольше — 50 ч, усреднение по 40–50 ч.
  - если разброс стал обычным — режим просто дольше устанавливался,
    новое значение затрат заменит старое (src/merge_results_v2.py);
  - если разброс остался большим — режим колебательный, точка
    исключается из модели по этому формальному критерию.

Каждая точка ~10 мин. Скрипт можно перезапускать: готовые пропускаются.

Запуск из корня проекта:  python src/rerun_unsteady.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import run_cost_experiment, save_histories   # noqa: E402
from src.cost_model_tools import unsteady_mask                     # noqa: E402

RESULTS = ROOT / 'data/doe_cost/runs/results_cost.csv'
OUT = ROOT / 'data/doe_cost/runs/results_cost_rerun.csv'
SP_COLS = ['ReactorTempSP', 'ReactorPressSP', 'ProductionSP',
           'ReactorLevelSP', 'AgitatorSpeedSP', 'SteamValvePosSP']
DURATION_H = 50.0
AVERAGE_FROM_H = 40.0
FACTOR = 2.0


def main():
    df = pd.read_csv(RESULTS)
    limit = FACTOR * df['cost_block_std'].median()
    bad = df[unsteady_mask(df, FACTOR)]
    print(f"Порог: σ блоков > {FACTOR} × медианы = {limit:.2f} $/ч")
    print(f"Не установились: {', '.join(str(int(e)) for e in bad['experiment_id']) or 'нет'}")

    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    rows = done.to_dict('records') if len(done) else []
    done_ids = set(done['experiment_id']) if len(done) else set()

    for _, r in bad.iterrows():
        exp_id = int(r['experiment_id'])
        if exp_id in done_ids:
            print(f"\nОпыт {exp_id}: уже перепрогнан, пропуск")
            continue
        sp = {c: float(r[c]) for c in SP_COLS}
        print(f"\n{'─' * 60}\n▶ Опыт {exp_id}: {DURATION_H:.0f} ч, усреднение {AVERAGE_FROM_H:.0f}–{DURATION_H:.0f} ч")
        t0 = time.time()
        res = run_cost_experiment(sp, duration=DURATION_H, t_from=AVERAGE_FROM_H)
        if res is None:
            print("   Не удалось, пропуск")
            continue
        save_histories(res, ROOT / f"data/doe_cost/runs/exp_{exp_id:03d}_long")
        new = res['row']
        steady = new['cost_block_std'] <= limit
        rows.append({'experiment_id': exp_id, 'point_type': r['point_type'], **new,
                     'duration_h': DURATION_H, 'average_from_h': AVERAGE_FROM_H,
                     'cost_old': r['operating_cost'], 'cost_block_std_old': r['cost_block_std'],
                     'steady_now': steady, 'elapsed_s': time.time() - t0})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"   было: {r['operating_cost']:.2f} $/ч (σ {r['cost_block_std']:.2f}); "
              f"стало: {new['operating_cost']:.2f} $/ч (σ {new['cost_block_std']:.2f}) → "
              + ('установился' if steady else 'КОЛЕБЛЕТСЯ — будет исключён'))

    if rows:
        out = pd.DataFrame(rows)
        print(f"\n{'=' * 60}\nИТОГ")
        print(out[['experiment_id', 'cost_old', 'cost_block_std_old', 'operating_cost',
                   'cost_block_std', 'steady_now']].round(2).to_string(index=False))
        print(f"Сохранено: {OUT}\nДальше: python src/merge_results_v2.py")


if __name__ == '__main__':
    main()
