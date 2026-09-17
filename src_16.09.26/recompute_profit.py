import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pandas as pd
import numpy as np
from src.main_function_1 import calc_tep_profit

df_results = pd.read_csv("data/doe_final/results.csv")

print(f"📋 Пересчёт прибыли для {len(df_results)} экспериментов\n")

profits = []
details = []

for idx, row in df_results.iterrows():
    exp_id = int(row['experiment_id'])
    exp_dir = Path(f"data/doe_final/exp_{exp_id:03d}")

    steady_xmeas = pd.read_csv(exp_dir / "steady_xmeas.csv").iloc[0].to_dict()
    steady_xmv = pd.read_csv(exp_dir / "steady_xmv.csv").iloc[0].to_dict()
    steady_data = {**steady_xmeas, **steady_xmv}

    profit, comp = calc_tep_profit(steady_data)
    profits.append(profit)
    details.append(comp)

    print(f"   Эксп. {exp_id:2d}: доход={comp['revenue']:7.2f}, "
          f"затраты={comp['total_cost']:7.2f}, прибыль={profit:7.2f}")

df_results['revenue'] = [d['revenue'] for d in details]
df_results['total_cost_profit'] = [d['total_cost'] for d in details]
df_results['profit'] = profits

df_results.to_csv("data/doe_final/results_profit.csv", index=False)

print(f"\n📊 Статистика прибыли:")
print(f"   Min:  {df_results['profit'].min():.2f}")
print(f"   Max:  {df_results['profit'].max():.2f}")
print(f"   Mean: {df_results['profit'].mean():.2f}")

print(f"\n🏆 Топ-5 по прибыли (максимум):")
top5 = df_results.nlargest(5, 'profit')[
    ['experiment_id', 'ReactorTempSP', 'ReactorPressSP', 'ProductionSP', 'profit']
]
print(top5.to_string(index=False))

print(f"\n💾 Сохранено в data/doe_final/results_profit.csv")