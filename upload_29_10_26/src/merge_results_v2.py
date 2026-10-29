# src/merge_results_v2.py
"""
Сборка уточнённого набора данных results_cost_v2.csv (секунды, без симулятора):
  1) исходные 47 опытов;
  2) опыты, перепрогнанные на 50 ч (rerun_unsteady.py): установившиеся
     заменяются новыми значениями, колеблющиеся исключаются;
  3) добавляются настоящие прогоны рядом с оптимумом: проверка оптимума
     (validation_cost.csv) и точки check_setpoint_effects.py.
     Это стандартное дополнение плана точками в районе оптимума
     (последовательный план): модель уточняется там, где была неуверенна.

Запуск из корня проекта:  python src/merge_results_v2.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

D = ROOT / 'data/doe_cost'
RESULTS = D / 'runs/results_cost.csv'
RERUN = D / 'runs/results_cost_rerun.csv'
VALIDATION = D / 'validation_cost.csv'
CHECKS = D / 'checks/setpoint_effects.csv'
OUT = D / 'runs/results_cost_v2.csv'


def main():
    df = pd.read_csv(RESULTS)
    cols = list(df.columns)
    df['source'] = 'plan_30h'
    print(f"Исходный план: {len(df)} опытов")

    if RERUN.exists():
        rr = pd.read_csv(RERUN)
        for r in rr.itertuples():
            i = df.index[df['experiment_id'] == r.experiment_id]
            if r.steady_now:
                for c in cols:
                    if c in rr.columns and c not in ('experiment_id', 'point_type'):
                        df.loc[i, c] = getattr(r, c)
                df.loc[i, 'source'] = 'rerun_50h'
                print(f"   опыт {r.experiment_id}: заменён прогоном 50 ч "
                      f"({r.cost_old:.2f} → {r.operating_cost:.2f} $/ч)")
            else:
                df = df.drop(i)
                print(f"   опыт {r.experiment_id}: исключён — колебательный режим "
                      f"(σ блоков {r.cost_block_std:.2f} $/ч и на 50 ч)")
    else:
        print("   перепрогонов нет (src/rerun_unsteady.py не запускался)")

    extra = []
    if VALIDATION.exists():
        v = pd.read_csv(VALIDATION).iloc[[0]].copy()
        v['experiment_id'], v['point_type'], v['source'] = 100, 'validation', 'validation_30h'
        extra.append(v)
    if CHECKS.exists():
        c = pd.read_csv(CHECKS).copy()
        c['source'] = 'check_30h'
        extra.append(c)
    for e in extra:
        e = e[[c for c in cols if c in e.columns] + ['source']]
        df = pd.concat([df, e], ignore_index=True)
    n_extra = sum(len(e) for e in extra)
    print(f"Добавлено точек у оптимума: {n_extra}")

    df = df.sort_values('experiment_id').reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nИтого: {len(df)} точек → {OUT}")
    print(df['source'].value_counts().to_string())
    print("\nДальше:\n   python src/robustness_cost.py --results data/doe_cost/runs/results_cost_v2.csv"
          "\n   python src/build_model_cost.py --results data/doe_cost/runs/results_cost_v2.csv --tag v2"
          "\n   python src/validate_cost_optimum.py --tag v2 --repeats 1")


if __name__ == '__main__':
    main()
