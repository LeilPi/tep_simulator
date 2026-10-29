# src/cost_common.py
"""
Общие функции для стандартной постановки TEP (минимизация эксплуатационных
затрат по Downs & Vogel, 1993, Table 9).

Один эксперимент = запуск симулятора через уже отлаженные функции
src/run_cube_5.py + расчёт затрат по стационарному участку 20–30 ч.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.run_cube_5 import run_experiment_with_retry          # noqa: E402
from src.tep_economics import calc_tep_operating_cost          # noqa: E402

# Уставки номинального режима Mode 1 (значения pyTEP по умолчанию)
NOMINAL_SP = {
    'ReactorTempSP': 122.9,
    'ReactorPressSP': 2800.0,
    'ProductionSP': 22.89,
    'ReactorLevelSP': 75.0,
    'AgitatorSpeedSP': 200.0,
    'SteamValvePosSP': 47.446,
}

# Варьируемые уставки (ProductionSP зафиксирована — стандартная постановка)
VARIABLES = ['ReactorTempSP', 'ReactorPressSP', 'ReactorLevelSP',
             'AgitatorSpeedSP', 'SteamValvePosSP']
FIXED = {'ProductionSP': 22.89}

DURATION_H = 30.0        # длительность прогона, ч
AVERAGE_FROM_H = 20.0    # усреднять с 20-го часа (10 ч стационарного участка)
BLOCK_H = 2.0            # блоки для оценки разброса внутри прогона
P_LIMIT_KPA = 2895.0     # рабочий предел давления в реакторе (Downs & Vogel)


def steady_average(df, t_from=AVERAGE_FROM_H, t_to=None):
    """Средние значения по участку t_from <= time <= t_to."""
    m = df['time'] >= t_from
    if t_to is not None:
        m &= df['time'] <= t_to
    return df.loc[m].mean(numeric_only=True)


def run_cost_experiment(sp_values, duration=DURATION_H, t_from=AVERAGE_FROM_H,
                        max_retries=3):
    """Прогон симулятора и расчёт затрат. None — если прогон не удался."""
    result = run_experiment_with_retry(sp_values=sp_values, duration=duration,
                                       max_retries=max_retries)
    if result is None:
        return None

    df_xmeas = result['df_xmeas']
    df_xmv = result['df_xmv']

    steady = steady_average(df_xmeas, t_from)
    cost, comp = calc_tep_operating_cost(steady)

    # Затраты по 2-часовым блокам — разброс внутри прогона (шум)
    block_costs = []
    t0 = t_from
    while t0 + BLOCK_H <= df_xmeas['time'].max() + 1e-9:
        block_costs.append(calc_tep_operating_cost(
            steady_average(df_xmeas, t0, t0 + BLOCK_H))[0])
        t0 += BLOCK_H

    p = df_xmeas['Reactor Pressure']
    out = {
        **sp_values,
        'operating_cost': cost,
        'purge_loss': comp['purge_loss'],
        'product_loss': comp['product_loss'],
        'compressor_cost': comp['compressor_cost'],
        'steam_cost': comp['steam_cost'],
        'product_molar_flow': comp['product_molar_flow'],
        'cost_block_std': float(np.std(block_costs, ddof=1)) if len(block_costs) > 1 else np.nan,
        'p_reactor_mean': float(steady['Reactor Pressure']),
        'p_reactor_max': float(p.max()),
        # среднее на стационаре выше рабочего предела или пики близко к останову (3000 кПа)
        'p_over_limit': bool(steady['Reactor Pressure'] > P_LIMIT_KPA + 1.0 or p.max() > 2950.0),
        'sim_end_time': float(df_xmeas['time'].max()),
        'completed': bool(df_xmeas['time'].max() >= duration - 0.06),
    }
    return {'row': out, 'df_xmeas': df_xmeas, 'df_xmv': df_xmv,
            'steady': steady, 'block_costs': block_costs}


def save_histories(res, exp_dir):
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    res['df_xmeas'].to_csv(exp_dir / 'xmeas_history.csv', index=False)
    res['df_xmv'].to_csv(exp_dir / 'xmv_history.csv', index=False)
    res['steady'].to_frame().T.to_csv(exp_dir / 'steady_xmeas.csv', index=False)
