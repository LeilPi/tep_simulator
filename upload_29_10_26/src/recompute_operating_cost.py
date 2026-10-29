"""
Пересчёт эксплуатационных затрат (Downs & Vogel, Table 9) для уже проведённых
экспериментов — без повторной симуляции, по сохранённым стационарным средним.

Запуск из корня репозитория:
    python src/recompute_operating_cost.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.tep_economics import calc_tep_operating_cost, to_dict  # noqa: E402

RUNS = ROOT / "data/doe_extended/runs"
steady = pd.read_csv(RUNS / "steady_data_extended.csv")
design = pd.read_csv(RUNS / "results_extended.csv").set_index("experiment_id")

rows = []
for _, row in steady.iterrows():
    cost, c = calc_tep_operating_cost(to_dict(row))
    rows.append({
        "experiment_id": int(row["experiment_id"]),
        "operating_cost": cost,
        "purge_loss": c["purge_loss"],
        "product_loss": c["product_loss"],
        "compressor_cost": c["compressor_cost"],
        "steam_cost": c["steam_cost"],
        "product_molar_flow": c["product_molar_flow"],
        "cost_per_kgmol_product": cost / c["product_molar_flow"],
    })

out = pd.DataFrame(rows).set_index("experiment_id").join(
    design.drop(columns=["profit", "is_steady", "elapsed_time"], errors="ignore"))
out.to_csv(RUNS / "results_operating_cost.csv")

print(out[["operating_cost", "purge_loss", "product_loss",
           "compressor_cost", "steam_cost"]].describe().round(2).T)
print(f"\nСохранено: {RUNS / 'results_operating_cost.csv'}")
