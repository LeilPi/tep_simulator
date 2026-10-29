# src/build_model_cost.py
"""
Суррогатная модель эксплуатационных затрат (5 уставок, ProductionSP фиксирована)
и поиск МИНИМУМА затрат.

Отличия от build_model_extended.py:
  - целевая переменная — operating_cost (Downs & Vogel, Table 9), минимизация;
  - масштабирование и полином внутри Pipeline — нет утечки данных в кросс-валидации;
  - alpha Ridge подбирается по кросс-валидации, а не задаётся 1.0;
  - модель сравнивается с простыми базовыми (среднее, линейная регрессия);
  - оптимизация стартует из нескольких лучших точек плана.

Запуск из корня проекта:
    python src/build_model_cost.py
    python src/build_model_cost.py --results data/doe_cost/runs/results_cost_v2.csv --tag v2
(с --tag результаты пишутся в model_cost_v2.pkl и cost_model_v2.png, первые не затираются)
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import VARIABLES                      # noqa: E402
from src.generate_hypercube_cost import BOUNDS             # noqa: E402

RESULTS = ROOT / 'data/doe_cost/runs/results_cost.csv'
NOMINAL = ROOT / 'data/doe_cost/nominal_cost.csv'


def cv_metrics(model, X, y, kf):
    pred = cross_val_predict(model, X, y, cv=kf)
    return r2_score(y, pred), np.sqrt(mean_squared_error(y, pred)), pred


def main(results=RESULTS, tag=''):
    suffix = f'_{tag}' if tag else ''
    MODEL_OUT = ROOT / f'data/doe_cost/model_cost{suffix}.pkl'
    FIG_OUT = ROOT / f'figures/cost_model{suffix}.png'
    df = pd.read_csv(results)
    if 'completed' in df:
        df = df[df['completed'].astype(bool)]
    X = df[VARIABLES].values
    y = df['operating_cost'].values
    print(f"Данных: {len(df)} экспериментов; затраты {y.min():.2f}–{y.max():.2f} $/ч")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # --- базовые модели для сравнения ---
    rmse_mean = np.sqrt(np.mean((y - y.mean()) ** 2))
    r2_lin, rmse_lin, _ = cv_metrics(make_pipeline(StandardScaler(), LinearRegression()), X, y, kf)

    # --- полином 2-й степени + Ridge, alpha по кросс-валидации ---
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge())
    grid = GridSearchCV(poly, {'ridge__alpha': [0.01, 0.1, 0.3, 1, 3, 10, 30]},
                        cv=kf, scoring='neg_mean_squared_error')
    grid.fit(X, y)
    alpha = grid.best_params_['ridge__alpha']
    poly.set_params(ridge__alpha=alpha)
    r2_poly, rmse_poly, pred_poly = cv_metrics(poly, X, y, kf)

    print("\nКАЧЕСТВО (5-fold кросс-валидация):")
    print(f"   среднее значение:          RMSE = {rmse_mean:6.2f} $/ч")
    print(f"   линейная регрессия:        R² = {r2_lin:6.3f}, RMSE = {rmse_lin:6.2f} $/ч")
    print(f"   полином 2 + Ridge (α={alpha}): R² = {r2_poly:6.3f}, RMSE = {rmse_poly:6.2f} $/ч")
    if rmse_poly >= rmse_lin:
        print("   ВНИМАНИЕ: полином не лучше линейной модели — проверьте данные")

    noise = None
    if NOMINAL.exists():
        nom = pd.read_csv(NOMINAL)
        noise = float(nom['cost_block_std'].mean())
        print(f"   шум симулятора (σ блоков номинала) ≈ {noise:.2f} $/ч")

    # --- обучение на всех данных и поиск минимума ---
    poly.fit(X, y)
    bounds = [BOUNDS[v] for v in VARIABLES]

    def f(x):
        return float(poly.predict(x.reshape(1, -1))[0])

    starts = list(df.nsmallest(5, 'operating_cost')[VARIABLES].values)
    starts.append(np.array([(lo + hi) / 2 for lo, hi in bounds]))
    best = None
    for x0 in starts:
        r = minimize(f, np.asarray(x0, float), method='L-BFGS-B', bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    x_opt, cost_pred = best.x, best.fun

    print("\nМИНИМУМ ЗАТРАТ ПО МОДЕЛИ:")
    for v, x, (lo, hi) in zip(VARIABLES, x_opt, bounds):
        mark = ''
        if abs(x - lo) < 0.02 * (hi - lo):
            mark = '← нижняя граница'
        elif abs(x - hi) < 0.02 * (hi - lo):
            mark = '← верхняя граница'
        print(f"   {v:16s} = {x:8.2f}   ({lo}–{hi}) {mark}")
    print(f"   Затраты (модель) = {cost_pred:.2f} $/ч")

    i_best = int(np.argmin(y))
    print("\nСРАВНЕНИЕ:")
    print(f"   лучшая точка плана: {y[i_best]:.2f} $/ч (эксп. {int(df.iloc[i_best]['experiment_id'])})")
    print(f"   оптимум модели:     {cost_pred:.2f} $/ч  (ошибка модели ~ ±{rmse_poly:.1f})")
    if NOMINAL.exists():
        print(f"   номинал:            {nom['operating_cost'].mean():.2f} $/ч")
    print("   Окончательный ответ даёт только проверка на симуляторе: src/validate_cost_optimum.py")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': poly, 'variables': VARIABLES, 'bounds': bounds, 'x_opt': x_opt,
                 'cost_pred': cost_pred, 'alpha': alpha, 'cv_r2': r2_poly, 'cv_rmse': rmse_poly,
                 'cv_rmse_linear': rmse_lin, 'noise': noise,
                 'best_doe': df.iloc[i_best][VARIABLES].to_dict(), 'best_doe_cost': float(y[i_best])},
                MODEL_OUT)
    print(f"\nМодель сохранена: {MODEL_OUT}")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, pred_poly, s=22)
    lims = [min(y.min(), pred_poly.min()) - 2, max(y.max(), pred_poly.max()) + 2]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlabel('Затраты на симуляторе, $/ч')
    ax.set_ylabel('Предсказание (кросс-валидация), $/ч')
    ax.set_title(f'Модель затрат: R² = {r2_poly:.3f}, RMSE = {rmse_poly:.2f} $/ч')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)
    print(f"График: {FIG_OUT}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    main(a.results, a.tag)
