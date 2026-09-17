import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

df = pd.read_csv("data/doe_final/results_accurate.csv")

feature_cols = ['ReactorTempSP', 'ReactorPressSP', 'ProductionSP']
X = df[feature_cols].values
y = df['total_cost_new'].values

print("="*70)
print("ПОСТРОЕНИЕ МОДЕЛИ НА РЕАЛИСТИЧНОЙ ФУНКЦИИ СТОИМОСТИ")
print("="*70)
print(f"\n📋 Данных: {len(df)} экспериментов")
print(f"   Стоимость: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

# ============================================================
# 2. МАСШТАБИРОВАНИЕ И РАЗДЕЛЕНИЕ
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================================
# 3. ОБУЧЕНИЕ (ПОЛИНОМ 2-Й СТЕПЕНИ)
# ============================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

print(f"\n📈 Качество модели:")
print(f"   R² (обучение): {r2_score(y_train, y_train_pred):.4f}")
print(f"   R² (тест):     {r2_score(y_test, y_test_pred):.4f}")
print(f"   RMSE (тест):   {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

# Кросс-валидация
cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
print(f"   CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 4. ПОИСК ОПТИМУМА
# ============================================================

def cost_function(x_scaled):
    x_poly = poly.transform(x_scaled.reshape(1, -1))
    return model.predict(x_poly)[0]

# Границы в масштабированном пространстве (по данным DOE)
bounds_scaled = [(X_scaled[:, i].min(), X_scaled[:, i].max()) for i in range(3)]

# Начальная точка — лучший эксперимент
best_idx = np.argmin(y)
x0 = X_scaled[best_idx]

result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds_scaled)

if result.success:
    x_opt_scaled = result.x
    x_opt = scaler.inverse_transform(x_opt_scaled.reshape(1, -1)).flatten()
    cost_opt = result.fun

    print(f"\n🏆 НОВЫЙ ОПТИМУМ (реалистичная функция):")
    print(f"   ReactorTempSP  = {x_opt[0]:.3f} °C")
    print(f"   ReactorPressSP = {x_opt[1]:.3f} кПа")
    print(f"   ProductionSP   = {x_opt[2]:.3f}")
    print(f"   Предсказанная стоимость = {cost_opt:.4f}")

# ============================================================
# 5. СРАВНЕНИЕ С DOE
# ============================================================

print(f"\n📊 Сравнение с экспериментами DOE:")
print(f"   Лучший эксперимент DOE: №{df.loc[best_idx, 'experiment_id']:.0f} ({y[best_idx]:.2f})")
print(f"   Оптимум модели:         {cost_opt:.2f}")
print(f"   Улучшение:              {y[best_idx] - cost_opt:.2f} ({100*(y[best_idx]-cost_opt)/y[best_idx]:.1f}%)")

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Факт vs предсказание
axes[0].scatter(y_train, y_train_pred, alpha=0.6, label='Обучение', color='blue')
axes[0].scatter(y_test, y_test_pred, alpha=0.8, label='Тест', color='red')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
axes[0].set_xlabel('Фактическая стоимость')
axes[0].set_ylabel('Предсказанная стоимость')
axes[0].set_title('Реалистичная функция стоимости')
axes[0].legend()
axes[0].grid(True)

# Зависимость стоимости от температуры (при фиксированных Press/Prod)
temp_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
grid = np.column_stack([
    temp_range,
    np.full(50, x_opt[1]),
    np.full(50, x_opt[2])
])
grid_scaled = scaler.transform(grid)
costs = model.predict(poly.transform(grid_scaled))

axes[1].plot(temp_range, costs, 'b-', linewidth=2)
axes[1].axvline(x_opt[0], color='r', linestyle='--', label=f'Оптимум: {x_opt[0]:.1f} °C')
axes[1].set_xlabel('ReactorTempSP, °C')
axes[1].set_ylabel('Стоимость, $/ч')
axes[1].set_title('Зависимость стоимости от температуры')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("figures/model_accurate.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Модель обучена и сохранена в figures/model_accurate.png")

# ============================================================
# 7. СОХРАНЕНИЕ МОДЕЛИ ДЛЯ ПРОВЕРКИ
# ============================================================

import joblib
joblib.dump({
    'model': model,
    'poly': poly,
    'scaler': scaler,
    'x_opt': x_opt,
    'cost_opt': cost_opt
}, "data/doe_final/surrogate_model.pkl")
print(f"✅ Модель сохранена в data/doe_final/surrogate_model.pkl")