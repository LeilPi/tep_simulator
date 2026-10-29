# src/run_nominal_cost.py
"""
Номинальный режим Mode 1 с новой функцией затрат: несколько повторов.

Даёт две вещи:
  1) точку сравнения для оптимума;
  2) оценку шума — разброс между повторами и между 2-часовыми блоками
     внутри прогона. Разница меньше этого разброса улучшением не считается.

Если pyTEP детерминирован (все повторы совпадают), шум оценивается по блокам.

Запуск из корня проекта:  python src/run_nominal_cost.py --repeats 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import NOMINAL_SP, run_cost_experiment, save_histories   # noqa: E402

OUT = ROOT / 'data/doe_cost/nominal_cost.csv'


def main(repeats=3):
    rows = []
    for i in range(1, repeats + 1):
        print(f"\n▶ Номинал, повтор {i}/{repeats}")
        res = run_cost_experiment(dict(NOMINAL_SP))
        if res is None:
            print("   Не удалось")
            continue
        save_histories(res, ROOT / f"data/doe_cost/nominal/run_{i:02d}")
        rows.append({'repeat': i, **res['row']})
        print(f"   затраты = {res['row']['operating_cost']:.2f} $/ч, "
              f"блоки по 2 ч: {np.round(res['block_costs'], 2)}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    c = df['operating_cost']
    print(f"\n{'=' * 60}\nНОМИНАЛ: среднее {c.mean():.2f} $/ч")
    if len(c) > 1:
        print(f"   разброс между повторами σ = {c.std(ddof=1):.3f} $/ч")
        if c.std(ddof=1) < 1e-9:
            print("   повторы совпадают: симулятор детерминирован, шум — по блокам")
    print(f"   разброс 2-ч блоков внутри прогона σ ≈ {df['cost_block_std'].mean():.3f} $/ч")
    print(f"   составляющие: продувка {df['purge_loss'].mean():.2f}, продукт {df['product_loss'].mean():.2f}, "
          f"компрессор {df['compressor_cost'].mean():.2f}, пар {df['steam_cost'].mean():.2f}")
    print(f"Сохранено: {OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    main(ap.parse_args().repeats)
