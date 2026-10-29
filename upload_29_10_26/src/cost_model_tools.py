# src/cost_model_tools.py
"""
Общие функции для суррогатной модели затрат: обучение, поиск минимума,
чувствительность. Та же логика, что в build_model_cost.py
(Pipeline: StandardScaler → PolynomialFeatures(2) → Ridge, alpha по 5-fold CV,
мультистарт L-BFGS-B), вынесенная в функции, чтобы сравнивать варианты данных.
"""

import sys
from pathlib import Path

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

ALPHAS = [0.01, 0.1, 0.3, 1, 3, 10, 30]
NOMINAL_CSV = ROOT / 'data/doe_cost/nominal_cost.csv'


def noise_sigma(default=None):
    """σ затрат по 2-часовым блокам номинала (оценка шума), $/ч."""
    if NOMINAL_CSV.exists():
        return float(pd.read_csv(NOMINAL_CSV)['cost_block_std'].mean())
    return default


def unsteady_mask(df, factor=2.0):
    """True для опытов, где разброс затрат по блокам > factor × медианы по плану."""
    s = df['cost_block_std']
    return s > factor * s.median()


def fit_and_optimize(df):
    """Обучает полином 2 + Ridge и ищет минимум. Возвращает словарь с результатами."""
    if 'completed' in df:
        df = df[df['completed'].astype(bool)]
    X = df[VARIABLES].values
    y = df['operating_cost'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    lin_pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=kf)
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge())
    grid = GridSearchCV(poly, {'ridge__alpha': ALPHAS}, cv=kf, scoring='neg_mean_squared_error')
    grid.fit(X, y)
    alpha = grid.best_params_['ridge__alpha']
    poly.set_params(ridge__alpha=alpha)
    pred = cross_val_predict(poly, X, y, cv=kf)
    poly.fit(X, y)

    bounds = [BOUNDS[v] for v in VARIABLES]

    def f(x):
        return float(poly.predict(np.asarray(x, float).reshape(1, -1))[0])

    starts = list(df.nsmallest(5, 'operating_cost')[VARIABLES].values)
    starts.append(np.array([(lo + hi) / 2 for lo, hi in bounds]))
    best = min((minimize(f, np.asarray(s, float), method='L-BFGS-B', bounds=bounds) for s in starts),
               key=lambda r: r.fun)
    return {
        'n': len(df), 'alpha': alpha, 'model': poly, 'f': f, 'bounds': bounds,
        'cv_r2': r2_score(y, pred), 'cv_rmse': float(np.sqrt(mean_squared_error(y, pred))),
        'cv_rmse_linear': float(np.sqrt(mean_squared_error(y, lin_pred))),
        'x_opt': best.x, 'cost_pred': float(best.fun),
    }


def sensitivity(res):
    """Затраты по модели, если в оптимуме сдвинуть одну уставку на нижнюю/верхнюю границу."""
    rows = []
    x = res['x_opt']
    base = res['f'](x)
    for i, (v, (lo, hi)) in enumerate(zip(VARIABLES, res['bounds'])):
        vals = {}
        for name, val in (('lo', lo), ('hi', hi)):
            z = x.copy()
            z[i] = val
            vals[name] = res['f'](z)
        rows.append({'variable': v, 'x_opt': x[i], 'cost_at_lo': vals['lo'], 'cost_at_hi': vals['hi'],
                     'max_increase': max(vals['lo'], vals['hi']) - base})
    return pd.DataFrame(rows)


def bound_mark(x, lo, hi):
    if abs(x - lo) < 0.02 * (hi - lo):
        return 'нижн.'
    if abs(x - hi) < 0.02 * (hi - lo):
        return 'верхн.'
    return 'внутри'
