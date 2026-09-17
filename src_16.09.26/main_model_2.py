import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
import joblib

# ============================================================
# 1. ЗАГРУЗКА
# ============================================================

df = pd.read_csv("data/doe_final/results_profit.csv")

feature_cols = ['ReactorTempSP', 'ReactorPressSP', 'ProductionSP']
X = df[feature_cols].values
y = df['profit'].values   # ← максимизируем прибыль

print("="*70)
print("МОДЕЛЬ ПРИБЫЛИ (max profit)")
print("="*70)
print(f"\n📋 Данных: {len(df)} экспериментов")
print(f"   Прибыль: min={y.min():.2f}, max={y.max():.2f}")

# ============================================================
# 2. МАСШТАБИРОВАНИЕ
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================================
# 3. ОБУЧЕНИЕ (Ridge для устойчивости)
# ============================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

print(f"\n📈 Качество модели:")
print(f"   R² (обучение): {r2_score(y_train, y_train_pred):.4f}")
print(f"   R² (тест):     {r2_score(y_test, y_test_pred):.4f}")
print(f"   RMSE (тест):   {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")

cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
print(f"   CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 4. ОПТИМИЗАЦИЯ (максимизация прибыли = минимизация -profit)
# ============================================================

def neg_profit(x_scaled):
    x_poly = poly.transform(x_scaled.reshape(1, -1))
    return -model.predict(x_poly)[0]

bounds_scaled = [(X_scaled[:, i].min(), X_scaled[:, i].max()) for i in range(3)]
x0 = X_scaled[np.argmax(y)]

result = minimize(neg_profit, x0, method='L-BFGS-B', bounds=bounds_scaled)

if result.success:
    x_opt = scaler.inverse_transform(result.x.reshape(1, -1)).flatten()
    profit_opt = -result.fun

    print(f"\n🏆 ОПТИМУМ (max прибыль):")
    print(f"   ReactorTempSP  = {x_opt[0]:.3f} °C")
    print(f"   ReactorPressSP = {x_opt[1]:.3f} кПа")
    print(f"   ProductionSP   = {x_opt[2]:.3f}")
    print(f"   Прибыль        = {profit_opt:.2f} $/ч")

    # Сравнение
    best_doe = y.max()
    best_idx = np.argmax(y)
    print(f"\n📊 Сравнение:")
    print(f"   Лучший DOE: №{df.loc[best_idx, 'experiment_id']:.0f} ({best_doe:.2f})")
    print(f"   Оптимум:    {profit_opt:.2f}")
    print(f"   Улучшение:  {profit_opt - best_doe:.2f} ({100*(profit_opt-best_doe)/abs(best_doe):.2f}%)")

# ============================================================
# 5. СОХРАНЕНИЕ
# ============================================================

joblib.dump({
    'model': model, 'poly': poly, 'scaler': scaler,
    'x_opt': x_opt, 'profit_opt': profit_opt,
    'feature_cols': feature_cols
}, "data/doe_final/model_profit.pkl")

print(f"\n✅ Модель сохранена в data/doe_final/model_profit.pkl")