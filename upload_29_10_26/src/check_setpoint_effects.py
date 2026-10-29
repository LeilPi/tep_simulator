# src/check_setpoint_effects.py
"""
Прямая проверка на симуляторе тех уставок, положение которых модель
определяет неуверенно (клапан пара и мешалка, см. src/robustness_cost.py).

Берётся проверенный оптимум (data/doe_cost/validation_cost.csv) и у него
меняется ОДНА уставка — остальные те же. Разница затрат с оптимумом и есть
влияние этой уставки, измеренное без модели:
    steam_25  — клапан пара 25 % вместо 0
    steam_50  — клапан пара 50 % вместо 0
    agit_250  — мешалка 250 об/мин вместо 200
Разница больше 2σ шума — влияние значимо; меньше — неразличимо.

pyTEP детерминирован, поэтому каждая точка считается один раз (~6 мин).
Скрипт можно перезапускать: готовые точки пропускаются.

Запуск из корня проекта:  python src/check_setpoint_effects.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.cost_common import VARIABLES, FIXED, run_cost_experiment, save_histories   # noqa: E402
from src.cost_model_tools import noise_sigma                                       # noqa: E402

VALIDATION = ROOT / 'data/doe_cost/validation_cost.csv'
OUT_DIR = ROOT / 'data/doe_cost/checks'
OUT = OUT_DIR / 'setpoint_effects.csv'

# имя точки: (id, что меняем, новое значение)
CHECKS = {
    'steam_25': (101, 'SteamValvePosSP', 25.0),
    'steam_50': (102, 'SteamValvePosSP', 50.0),
    'agit_250': (103, 'AgitatorSpeedSP', 250.0),
}
COMPONENTS = ['purge_loss', 'product_loss', 'compressor_cost', 'steam_cost']


def main():
    ref = pd.read_csv(VALIDATION).iloc[0]
    base_sp = {v: float(ref[v]) for v in VARIABLES}
    base_sp.update(FIXED)
    print("Опорная точка — проверенный оптимум:")
    for k, v in base_sp.items():
        print(f"   {k:16s} = {v:9.3f}")
    print(f"   затраты = {ref['operating_cost']:.2f} $/ч")

    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    rows = done.to_dict('records') if len(done) else []
    done_names = set(done['check']) if len(done) else set()

    for name, (exp_id, var, value) in CHECKS.items():
        if name in done_names:
            print(f"\n{name}: уже есть, пропуск")
            continue
        sp = dict(base_sp)
        sp[var] = value
        print(f"\n{'─' * 60}\n▶ {name}: {var} = {value} (в оптимуме {base_sp[var]:.2f})")
        t0 = time.time()
        res = run_cost_experiment(sp)
        if res is None:
            print("   Не удалось, пропуск")
            continue
        save_histories(res, OUT_DIR / name)
        rows.append({'check': name, 'experiment_id': exp_id, 'point_type': 'check',
                     'changed': var, **res['row'], 'elapsed_s': time.time() - t0})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"   затраты = {res['row']['operating_cost']:.2f} $/ч")

    if not rows:
        return
    df = pd.DataFrame(rows)
    sigma = noise_sigma(default=float(ref['cost_block_std']))
    print(f"\n{'=' * 72}\nВЛИЯНИЕ УСТАВОК (симулятор, без модели); 2σ шума = {2 * sigma:.2f} $/ч")
    print(f"{'точка':10s} {'затраты':>8s} {'Δ':>7s}   "
          + '  '.join(f'{c[:9]:>9s}' for c in COMPONENTS) + '   вывод')
    print(f"{'оптимум':10s} {ref['operating_cost']:8.2f} {0:7.2f}   "
          + '  '.join(f"{ref[c]:9.2f}" for c in COMPONENTS))
    for r in df.itertuples():
        d = r.operating_cost - ref['operating_cost']
        if d > 2 * sigma:
            verdict = 'хуже оптимума, значимо'
        elif d < -2 * sigma:
            verdict = 'ЛУЧШЕ оптимума, значимо — оптимум неверен'
        else:
            verdict = 'неразличимо (< 2σ)'
        print(f"{r.check:10s} {r.operating_cost:8.2f} {d:+7.2f}   "
              + '  '.join(f"{getattr(r, c):9.2f}" for c in COMPONENTS) + f'   {verdict}')
    print("\nΔ по составляющим показывает, ЗА СЧЁТ ЧЕГО меняются затраты "
          "(например, пар дороже, но меньше потери продукта).")
    print(f"Сохранено: {OUT}")


if __name__ == '__main__':
    main()
