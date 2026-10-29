# src/robustness_cost.py
"""
Проверка устойчивости вывода об оптимуме (без новых прогонов симулятора, секунды).

Модель строится на четырёх вариантах данных:
  1) все точки плана;
  2) без якорных точек (43–47) — ответ на вопрос «не подогнан ли оптимум якорями»;
  3) без неустановившихся опытов (разброс блоков > 2 × медианы);
  4) без якорей и без неустановившихся.
Для каждого варианта печатаются качество, точка минимума и чувствительность:
насколько вырастут затраты по модели, если одну уставку сдвинуть на границу.
Сдвиг меньше 2σ шума — влияние уставки неразличимо.

Запуск из корня проекта:
    python src/robustness_cost.py
    python src/robustness_cost.py --results data/doe_cost/runs/results_cost_v2.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import VARIABLES                                     # noqa: E402
from src.cost_model_tools import (bound_mark, fit_and_optimize, noise_sigma,   # noqa: E402
                                  sensitivity, unsteady_mask)

DEFAULT_RESULTS = ROOT / 'data/doe_cost/runs/results_cost.csv'
OUT = ROOT / 'data/doe_cost/checks/robustness.csv'
SHORT = {'ReactorTempSP': 'T', 'ReactorPressSP': 'P', 'ReactorLevelSP': 'L',
         'AgitatorSpeedSP': 'A', 'SteamValvePosSP': 'S'}


def main(results):
    df = pd.read_csv(results)
    sigma = noise_sigma()
    unsteady = unsteady_mask(df)
    anchor = df['point_type'].eq('anchor')
    print(f"Данные: {results.name}, {len(df)} точек")
    if unsteady.any():
        print("Неустановившиеся (σ блоков > 2 × медианы "
              f"{df['cost_block_std'].median():.2f}): опыты "
              + ', '.join(f"{int(e)} (σ={s:.1f})" for e, s in
                          df.loc[unsteady, ['experiment_id', 'cost_block_std']].values))
    if sigma:
        print(f"Шум σ = {sigma:.2f} $/ч, порог различимости 2σ = {2 * sigma:.2f} $/ч")

    variants = [('все точки', df),
                ('без якорей', df[~anchor]),
                ('без неустановившихся', df[~unsteady]),
                ('без якорей и неустановившихся', df[~anchor & ~unsteady])]

    rows = []
    fit_bounds = None
    for name, d in variants:
        r = fit_and_optimize(d)
        sens = sensitivity(r)
        fit_bounds = r['bounds']
        print(f"\n{'─' * 70}\n{name.upper()}: n = {r['n']}, α = {r['alpha']}, "
              f"R² = {r['cv_r2']:.3f}, RMSE = {r['cv_rmse']:.2f} (линейная {r['cv_rmse_linear']:.2f}) $/ч")
        print(f"   минимум по модели: {r['cost_pred']:.2f} $/ч")
        print(f"   {'уставка':16s} {'оптимум':>9s}  граница   затраты при сдвиге на границу (мин / макс)  вывод")
        for (v, (lo, hi)), x, s in zip(zip(VARIABLES, r['bounds']), r['x_opt'], sens.itertuples()):
            verdict = ''
            if sigma:
                verdict = 'значимо' if s.max_increase > 2 * sigma else 'неразличимо (< 2σ)'
            print(f"   {v:16s} {x:9.2f}  {bound_mark(x, lo, hi):7s}  "
                  f"{s.cost_at_lo:7.2f} / {s.cost_at_hi:7.2f}  (+{s.max_increase:5.2f})     {verdict}")
        row = {'variant': name, 'n': r['n'], 'alpha': r['alpha'], 'cv_r2': r['cv_r2'],
               'cv_rmse': r['cv_rmse'], 'cv_rmse_linear': r['cv_rmse_linear'], 'cost_pred': r['cost_pred']}
        for v, x, s in zip(VARIABLES, r['x_opt'], sens.itertuples()):
            row[f'{SHORT[v]}_opt'] = x
            row[f'{SHORT[v]}_max_increase'] = s.max_increase
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\n{'=' * 70}\nСВОДКА: где оказался оптимум в каждом варианте")
    print(out[['variant', 'n', 'cv_rmse'] + [f'{SHORT[v]}_opt' for v in VARIABLES] + ['cost_pred']]
          .round(2).to_string(index=False))
    stable, unstable = [], []
    for v, (lo, hi) in zip(VARIABLES, fit_bounds):
        xs = out[f'{SHORT[v]}_opt']
        marks = {bound_mark(x, lo, hi) for x in xs}
        desc = f"{SHORT[v]} ({'/'.join(sorted(marks))}, {xs.min():.1f}–{xs.max():.1f})"
        (stable if len(marks) == 1 else unstable).append(desc)
    print(f"\nПоложение не зависит от выбора данных: {', '.join(stable) or '—'}")
    print(f"Зависит от выбора данных:              {', '.join(unstable) or '—'}")
    if unstable:
        print("   → эти уставки проверить прямо на симуляторе: python src/check_setpoint_effects.py")
    print(f"Сохранено: {OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    main(ap.parse_args().results)
