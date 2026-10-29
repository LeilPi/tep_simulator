# src/generate_hypercube_cost.py
"""
План эксперимента для стандартной постановки TEP:
ProductionSP зафиксирована (22,89), варьируются 5 уставок.

Состав плана (~47 точек):
  - 40 точек латинского гиперкуба;
  - центр области;
  - номинальный режим Mode 1;
  - 5 «якорных» точек в области оптимума Рикера (1995): давление у рабочего
    предела, уровень минимальный, мешалка максимальная, клапан пара закрыт.

Границы включают номинал, чтобы сравнение «оптимум vs номинал» шло внутри
области, где модель обучена.

Запуск из корня проекта:  python src/generate_hypercube_cost.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import VARIABLES, FIXED, NOMINAL_SP    # noqa: E402

BOUNDS = {
    'ReactorTempSP':   (119.0, 127.0),    # номинал 122,9
    'ReactorPressSP':  (2700.0, 2895.0),  # 2895 — рабочий предел (Downs & Vogel)
    'ReactorLevelSP':  (50.0, 75.0),      # номинал 75; 50 — нижний рабочий предел
    'AgitatorSpeedSP': (200.0, 250.0),    # номинал 200
    'SteamValvePosSP': (0.0, 50.0),       # номинал 47,4; у Рикера клапан закрыт
}

ANCHORS = [
    # T,     P,      Level, Agit,  Steam
    [122.0, 2895.0, 50.0, 250.0, 0.0],
    [124.0, 2895.0, 50.0, 250.0, 0.0],
    [122.0, 2850.0, 50.0, 250.0, 0.0],
    [122.0, 2895.0, 60.0, 250.0, 0.0],
    [122.0, 2895.0, 50.0, 225.0, 0.0],
]


def generate(n_lhs=40, seed=42, out=ROOT / 'data/doe_cost/design/hypercube_cost.csv'):
    lows = np.array([BOUNDS[v][0] for v in VARIABLES])
    highs = np.array([BOUNDS[v][1] for v in VARIABLES])

    X_lhs = lows + qmc.LatinHypercube(d=len(VARIABLES), seed=seed).random(n_lhs) * (highs - lows)
    X_center = ((lows + highs) / 2).reshape(1, -1)
    X_nominal = np.array([[NOMINAL_SP[v] for v in VARIABLES]])
    X_anchor = np.array(ANCHORS)

    X = np.clip(np.vstack([X_lhs, X_center, X_nominal, X_anchor]), lows, highs)
    kind = ['lhs'] * n_lhs + ['center', 'nominal'] + ['anchor'] * len(ANCHORS)

    df = pd.DataFrame(X, columns=VARIABLES)
    for k, v in FIXED.items():
        df[k] = v
    df.insert(0, 'experiment_id', range(1, len(df) + 1))
    df['point_type'] = kind

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"План сохранён: {out}  ({len(df)} точек)")
    print(df.describe().loc[['min', 'max']].round(2).to_string())
    return df


if __name__ == '__main__':
    generate()
