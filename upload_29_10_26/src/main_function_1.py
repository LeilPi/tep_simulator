import numpy as np
import pandas as pd


def calc_tep_cost(data, st_state_window=20):
    """Экономическая стоимость TEP (затраты)."""
    # ... (оставляем как есть)
    pass


def calc_tep_profit(data, st_state_window=20):
    """
    Прибыль TEP = Доход от продуктов − Затраты.

    Доход: G = 30.44 $/кгмоль, H = 22.94 $/кгмоль.
    Затраты: см. calc_tep_cost.
    """
    # === Усреднение ===
    if isinstance(data, pd.DataFrame):
        if len(data) > st_state_window:
            data = data.iloc[-st_state_window:].mean()
        else:
            data = data.mean()
        data_dict = data.to_dict()
    elif isinstance(data, pd.Series):
        data_dict = data.to_dict()
    else:
        data_dict = data

    # === ЗАТРАТЫ ===
    costs = {
        'A': 2.206, 'C': 6.177, 'D': 22.06, 'E': 14.56,
        'F': 17.89, 'G': 30.44, 'H': 22.94
    }

    # Продувка
    purge_cost_per_kgmol = 0.0
    for comp, cost in costs.items():
        key = f'Component {comp} in Purge'
        if key in data_dict:
            purge_cost_per_kgmol += (data_dict[key] / 100.0) * cost

    purge_rate = data_dict.get('Purge Rate', 0.0)
    purge_molar_flow = purge_rate * 1000.0 / 22.414
    purge_loss = purge_cost_per_kgmol * purge_molar_flow

    # Продукт (потери сырья D, E, F)
    product_cost_per_kgmol = 0.0
    for comp in ['D', 'E', 'F']:
        key = f'Component {comp} in Product'
        if key in data_dict:
            product_cost_per_kgmol += (data_dict[key] / 100.0) * costs[comp]

    product_rate = data_dict.get('Stripper Underflow', 0.0)
    product_molar_flow = product_rate * 850.0 / 70.0
    product_loss = product_cost_per_kgmol * product_molar_flow

    # Компрессор и пар
    compressor_cost = 0.0536 * data_dict.get('Compressor Work', 0.0)
    steam_cost = 0.0318 * data_dict.get('Stripper Steam Flow', 0.0)

    total_cost = purge_loss + product_loss + compressor_cost + steam_cost

    # === ДОХОД ===
    revenue = 0.0
    for comp, price in [('G', 30.44), ('H', 22.94)]:
        key = f'Component {comp} in Product'
        if key in data_dict:
            # Мол. доля (0-1) × мольный расход × цена
            mol_frac = data_dict[key] / 100.0
            revenue += mol_frac * product_molar_flow * price

    # === ПРИБЫЛЬ ===
    profit = revenue - total_cost

    components = {
        'revenue': revenue,
        'purge_loss': purge_loss,
        'product_loss': product_loss,
        'compressor_cost': compressor_cost,
        'steam_cost': steam_cost,
        'total_cost': total_cost,
        'profit': profit,
    }

    return profit, components

    return profit, components

    def calc_cost_from_files(xmeas_path, xmv_path, st_state_window=20):
        """Загружает данные из CSV и рассчитывает стоимость."""
        df_xmeas = pd.read_csv(xmeas_path)
        df_xmv = pd.read_csv(xmv_path)

        df_combined = pd.merge(df_xmeas, df_xmv, on='time', how='outer')

        if 'disturbance_id_x' in df_combined.columns and 'disturbance_id_y' in df_combined.columns:
            df_combined['disturbance_id'] = df_combined['disturbance_id_x'].fillna(0) + df_combined[
                'disturbance_id_y'].fillna(0)
            df_combined = df_combined.drop(columns=['disturbance_id_x', 'disturbance_id_y'])

        profit, components = calc_tep_cost(df_combined, st_state_window)
        return profit, components, df_combined



