import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc


def generate_hypercube_with_anchors(n_lhs=60, seed=42):
    """
    Комбинированный план:
    - 60 точек LHS (равномерное покрытие)
    - 8 угловых точек (границы)
    - 1 центральная точка
    - 6 «якорных» точек на оптимуме по литературе
    Итого: ~75 экспериментов.
    """
    sp_names = [
        'ReactorTempSP',
        'ReactorPressSP',
        'ProductionSP',
        'ReactorLevelSP',
        'AgitatorSpeedSP',
        'SteamValvePosSP',
    ]

    bounds = [
        (119.0, 127.0),
        (2700.0, 2895.0),
        (18.0, 24.0),
        (50.0, 65.0),
        (200.0, 250.0),
        (0.0, 15.0),
    ]

    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    # === 1. LHS ===
    sampler = qmc.LatinHypercube(d=len(sp_names), seed=seed)
    lhs_samples = sampler.random(n=n_lhs)
    X_lhs = lows + lhs_samples * (highs - lows)

    # === 2. Угловые точки (2^6 = 64, но возьмём выборочно) ===
    # Возьмём 8 углов: комбинации min/max по Temp, Press, Prod
    corners = []
    for t in [lows[0], highs[0]]:
        for p in [lows[1], highs[1]]:
            for prod in [lows[2], highs[2]]:
                corners.append([
                    t, p, prod,
                    lows[3],  # Level = min (по Ricker)
                    highs[4],  # Agitator = max
                    lows[5],  # Steam = min
                ])
    X_corners = np.array(corners)

    # === 3. Центральная точка ===
    X_center = ((lows + highs) / 2).reshape(1, -1)

    # === 4. «Якорные» точки на оптимуме по литературе ===
    anchors = [
        # Оптимум Ricker (1995):
        [122.0, 2895.0, 23.0, 50.0, 250.0, 0.0],
        # Близкие вариации:
        [120.0, 2895.0, 23.0, 50.0, 250.0, 0.0],
        [124.0, 2895.0, 23.0, 50.0, 250.0, 0.0],
        [122.0, 2800.0, 23.0, 50.0, 250.0, 0.0],
        [122.0, 2895.0, 22.0, 50.0, 250.0, 0.0],
        [122.0, 2895.0, 24.0, 50.0, 250.0, 0.0],
    ]
    X_anchors = np.array(anchors)

    # === Объединяем ===
    X_all = np.vstack([X_lhs, X_corners, X_center, X_anchors])

    # Обрезаем по границам
    X_all = np.clip(X_all, lows, highs)

    df = pd.DataFrame(X_all, columns=sp_names)
    df.insert(0, 'experiment_id', range(1, len(df) + 1))

    save_dir = Path("data/doe_extended/design")
    save_dir.mkdir(parents=True, exist_ok=True)
    design_path = save_dir / "hypercube_extended_v2.csv"
    df.to_csv(design_path, index=False)

    print(f"✅ Комбинированный план сохранён: {design_path}")
    print(f"   Всего точек: {len(df)}")
    print(f"      - LHS:      {n_lhs}")
    print(f"      - Угловые:  {len(X_corners)}")
    print(f"      - Центр:    1")
    print(f"      - Якорные:  {len(X_anchors)}")
    print(f"\n📊 Диапазоны ProductionSP в плане:")
    print(f"   Min: {df['ProductionSP'].min():.2f}")
    print(f"   Max: {df['ProductionSP'].max():.2f}")
    print(f"\n📊 Якорные точки (первые 6):")
    print(df.tail(6).to_string(index=False))

    return df


if __name__ == "__main__":
    generate_hypercube_with_anchors(n_lhs=60)