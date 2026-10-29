# src/validate_cost_optimum.py
"""
Проверка найденного минимума затрат на симуляторе (несколько повторов)
и сравнение с номиналом и лучшей точкой плана.

Запуск из корня проекта:
    python src/validate_cost_optimum.py --repeats 3
    python src/validate_cost_optimum.py --tag v2 --repeats 1
(pyTEP детерминирован — повторы совпадают, поэтому достаточно --repeats 1;
 с --tag берётся model_cost_v2.pkl и пишется validation_cost_v2.csv)
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import FIXED, run_cost_experiment, save_histories   # noqa: E402

NOMINAL = ROOT / 'data/doe_cost/nominal_cost.csv'


def main(repeats=3, tag=''):
    suffix = f'_{tag}' if tag else ''
    MODEL = ROOT / f'data/doe_cost/model_cost{suffix}.pkl'
    OUT = ROOT / f'data/doe_cost/validation_cost{suffix}.csv'
    m = joblib.load(MODEL)
    sp = {v: float(x) for v, x in zip(m['variables'], m['x_opt'])}
    sp.update(FIXED)
    print("Оптимум модели:")
    for k, v in sp.items():
        print(f"   {k:16s} = {v:9.3f}")
    print(f"   предсказанные затраты = {m['cost_pred']:.2f} $/ч")

    rows = []
    for i in range(1, repeats + 1):
        print(f"\n▶ Проверка, повтор {i}/{repeats}")
        res = run_cost_experiment(dict(sp))
        if res is None:
            print("   Не удалось")
            continue
        save_histories(res, ROOT / f"data/doe_cost/validation{suffix}/run_{i:02d}")
        rows.append({'repeat': i, **res['row'], 'cost_pred': m['cost_pred']})
        print(f"   затраты = {res['row']['operating_cost']:.2f} $/ч"
              + ("  ВНИМАНИЕ: давление выше 2895 кПа" if res['row']['p_over_limit'] else ''))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    c = df['operating_cost']
    noise = df['cost_block_std'].mean()

    print(f"\n{'=' * 60}\nРЕЗУЛЬТАТ")
    print(f"   оптимум (симулятор): {c.mean():.2f} $/ч"
          + (f" ± {c.std(ddof=1):.2f}" if len(c) > 1 else ''))
    print(f"   ошибка модели:       {c.mean() - m['cost_pred']:+.2f} $/ч")
    print(f"   лучшая точка плана:  {m['best_doe_cost']:.2f} $/ч")
    if NOMINAL.exists():
        nom = pd.read_csv(NOMINAL)['operating_cost']
        d = nom.mean() - c.mean()
        print(f"   номинал:             {nom.mean():.2f} $/ч")
        print(f"   снижение затрат:     {d:.2f} $/ч ({100 * d / nom.mean():.1f} %)")
        print(f"   шум (σ 2-ч блоков):  {noise:.2f} $/ч")
        if abs(d) < 2 * noise:
            print("   Разница меньше 2σ шума — улучшение не подтверждено")
    gain_vs_doe = m['best_doe_cost'] - c.mean()
    print(f"   оптимум лучше лучшей точки плана на {gain_vs_doe:+.2f} $/ч"
          + ("" if gain_vs_doe > 0 else "  — суррогат не дал выигрыша над планом"))
    print(f"Сохранено: {OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    main(a.repeats, a.tag)
