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
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

results_path = "data/doe_extended/runs/results_extended.csv"
df = pd.read_csv(results_path)

print("=" * 70)
print("ОБУЧЕНИЕ РАСШИРЕННОЙ МОДЕЛИ ПРИБЫЛИ (6 переменных)")
print("=" * 70)
print(f"\n📋 Данных: {len(df)} экспериментов")
print(f"   Колонки: {df.columns.tolist()}")

# ============================================================
# 2. ОПРЕДЕЛЯЕМ ПРИЗНАКИ И ЦЕЛЕВУЮ ПЕРЕМЕННУЮ
# ============================================================

feature_cols = [
    'ReactorTempSP',
    'ReactorPressSP',
    'ProductionSP',
    'ReactorLevelSP',
    'AgitatorSpeedSP',
    'SteamValvePosSP',
]

# Проверяем, что все колонки есть
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"\n❌ Отсутствуют колонки: {missing}")
    print(f"   Доступные: {df.columns.tolist()}")
    sys.exit(1)

X = df[feature_cols].values
y = df['profit'].values

print(f"\n📊 Целевая переменная (прибыль):")
print(f"   min  = {y.min():.2f}")
print(f"   max  = {y.max():.2f}")
print(f"   mean = {y.mean():.2f}")

# ============================================================
# 3. МАСШТАБИРОВАНИЕ
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n📊 Разделение данных:")
print(f"   Train: {len(X_train)}")
print(f"   Test:  {len(X_test)}")

# ============================================================
# 4. ОБУЧЕНИЕ (ПОЛИНОМ 2-Й СТЕПЕНИ + RIDGE)
# ============================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Считаем, сколько признаков получилось
n_features = X_train_poly.shape[1]
print(f"\n📊 Полиномиальные признаки: {n_features}")

model = Ridge(alpha=1.0)
model.fit(X_train_poly, y_train)

# Предсказания
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# ============================================================
# 5. МЕТРИКИ КАЧЕСТВА
# ============================================================

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n📈 КАЧЕСТВО МОДЕЛИ:")
print(f"   R² (обучение):  {r2_train:.4f}")
print(f"   R² (тест):      {r2_test:.4f}")
print(f"   RMSE (обучение): {rmse_train:.2f} $/ч")
print(f"   RMSE (тест):     {rmse_test:.2f} $/ч")

# Кросс-валидация
cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
print(f"   CV R² (5-fold):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 6. ОПТИМИЗАЦИЯ (МАКСИМИЗАЦИЯ ПРИБЫЛИ)
# ============================================================

def neg_profit(x_scaled):
    """Отрицательная прибыль для minimize."""
    x_poly = poly.transform(x_scaled.reshape(1, -1))
    return -model.predict(x_poly)[0]

# Границы — по данным DOE
bounds_scaled = [(X_scaled[:, i].min(), X_scaled[:, i].max()) for i in range(X_scaled.shape[1])]

# Начальная точка — лучший эксперимент
best_idx = np.argmax(y)
x0 = X_scaled[best_idx]

print(f"\n🎯 СТАРТ ОПТИМИЗАЦИИ:")
print(f"   Начальная прибыль: {y[best_idx]:.2f} $/ч")

result = minimize(neg_profit, x0, method='L-BFGS-B', bounds=bounds_scaled)

if result.success:
    x_opt_scaled = result.x
    x_opt = scaler.inverse_transform(x_opt_scaled.reshape(1, -1)).flatten()
    profit_opt = -result.fun

    print(f"\n🏆 ОПТИМУМ (расширенное пространство):")
    print(f"   {'─' * 50}")
    for i, name in enumerate(feature_cols):
        print(f"   {name:20s} = {x_opt[i]:8.3f}")
    print(f"   {'─' * 50}")
    print(f"   Прибыль = {profit_opt:.2f} $/ч")

    # Сравнение
    print(f"\n📊 СРАВНЕНИЕ:")
    print(f"   Лучший DOE:      {y.max():.2f} $/ч")
    print(f"   Оптимум модели:  {profit_opt:.2f} $/ч")
    print(f"   Прирост:         {profit_opt - y.max():+.2f} $/ч")
    print(f"   Прирост %:       {100 * (profit_opt - y.max()) / y.max():+.2f}%")

    # Сравнение с номиналом
    NOMINAL_PROFIT = 7225.26
    print(f"\n   Номинал (Mode 1): {NOMINAL_PROFIT:.2f} $/ч")
    print(f"   Прирост к номиналу: {profit_opt - NOMINAL_PROFIT:+.2f} $/ч")
    print(f"   Прирост %:          {100 * (profit_opt - NOMINAL_PROFIT) / NOMINAL_PROFIT:+.2f}%")

    # Проверка границ
    print(f"\n📊 Проверка границ (оптимум на границе?):")
    bounds_orig = [
        (119.0, 127.0),
        (2700.0, 2895.0),
        (18.0, 24.0),
        (50.0, 65.0),
        (200.0, 250.0),
        (0.0, 15.0),
    ]
    for i, (name, (low, high)) in enumerate(zip(feature_cols, bounds_orig)):
        val = x_opt[i]
        at_bound = ""
        if abs(val - low) < 0.1 * (high - low):
            at_bound = "← МИНИМУМ"
        elif abs(val - high) < 0.1 * (high - low):
            at_bound = "← МАКСИМУМ"
        print(f"   {name:20s} = {val:8.3f}  ({low:.1f} – {high:.1f}) {at_bound}")

    # Сохранение модели
    save_path = "data/doe_extended/model_profit_extended.pkl"
    joblib.dump({
        'model': model,
        'poly': poly,
        'scaler': scaler,
        'x_opt': x_opt,
        'profit_opt': profit_opt,
        'feature_cols': feature_cols,
        'bounds': bounds_orig,
    }, save_path)
    print(f"\n💾 Модель сохранена в {save_path}")

else:
    print(f"\n❌ Оптимизация не сошлась: {result.message}")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Факт vs предсказание
axes[0].scatter(y_train, y_train_pred, alpha=0.6, label='Обучение', color='blue')
axes[0].scatter(y_test, y_test_pred, alpha=0.8, label='Тест', color='red')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
axes[0].set_xlabel('Фактическая прибыль, $/ч')
axes[0].set_ylabel('Предсказанная прибыль, $/ч')
axes[0].set_title(f'R² тест = {r2_test:.3f}')
axes[0].legend()
axes[0].grid(True)

# Распределение прибыли в DOE
axes[1].hist(y, bins=15, edgecolor='black', alpha=0.7)
axes[1].axvline(y.max(), color='green', linestyle='--', label=f'Лучший DOE = {y.max():.0f}')
axes[1].axvline(profit_opt, color='red', linestyle='--', label=f'Оптимум = {profit_opt:.0f}')
axes[1].axvline(NOMINAL_PROFIT, color='blue', linestyle=':', label=f'Номинал = {NOMINAL_PROFIT:.0f}')
axes[1].set_xlabel('Прибыль, $/ч')
axes[1].set_ylabel('Частота')
axes[1].set_title('Распределение прибыли в DOE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/extended_model.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Графики сохранены в figures/extended_model.png")
print(f"\n{'=' * 70}")
print("ГОТОВО! Следующий шаг — проверка оптимума на симуляторе.")
print(f"{'=' * 70}")