"""
Экономика Tennessee Eastman Process.

1) calc_tep_operating_cost — СТАНДАРТНАЯ функция эксплуатационных затрат
   по Downs & Vogel (1993), Table 9. Минимизируется. Контрольное значение
   для базового режима: 170,6 $/ч (продувка 114,7 + потери D, E, F с продуктом
   30,3 + компрессор 18,3 + пар 7,3).

2) calc_tep_profit_extended — АВТОРСКОЕ РАСШИРЕНИЕ (не перенос Table 9):
   выручка от G и H минус эксплуатационные затраты. Цены G и H задаются
   явно при вызове; значения 30,44 и 22,94 из Table 9 — это стоимости
   компонентов для оценки потерь, а не цены продажи.

Совместимо с Python 3.7. Вход — dict / pandas.Series / pandas.DataFrame
со средними стационарными значениями XMEAS. Понимаются имена столбцов pyTEP
('Purge Rate', 'Component A in Purge', ...) в том числе с префиксом 'xmeas_'.
"""

import math

# ------------------------------------------------------------------
# Константы Downs & Vogel (1993), Table 9
# ------------------------------------------------------------------

# Стоимость компонентов, $/кгмоль (B — инерт, не оценивается)
COMPONENT_COST = {'A': 2.206, 'C': 6.177, 'D': 22.06, 'E': 14.56,
                  'F': 17.89, 'G': 30.44, 'H': 22.94}

PURGE_KGMOL_PER_KSCM = 44.79     # кгмоль/ч на 1 kscmh (пересчёт XMEAS 10)
PRODUCT_KGMOL_PER_M3 = 9.21      # кгмоль/м3 (базовый режим: 211,3 / 22,949)
COMPRESSOR_COST = 0.0536         # $/кВт·ч  (XMEAS 20, кВт)
STEAM_COST = 0.0318              # $/кг     (XMEAS 19, кг/ч)

PURGE_COMPONENTS = ['A', 'C', 'D', 'E', 'F', 'G', 'H']   # XMEAS 29, 31–36
PRODUCT_LOSS_COMPONENTS = ['D', 'E', 'F']                # XMEAS 37–39

# Имена переменных pyTEP
NAMES = {
    'purge_rate': 'Purge Rate',                  # XMEAS 10, kscmh
    'product_rate': 'Stripper Underflow',        # XMEAS 17, м3/ч
    'stripper_temp': 'Stripper Temp',            # XMEAS 18, °C
    'steam': 'Stripper Steam Flow',              # XMEAS 19, кг/ч
    'compressor': 'Compressor Work',             # XMEAS 20, кВт
}


def purge_key(c):
    return 'Component {} in Purge'.format(c)     # XMEAS 29–36, мол.%


def product_key(c):
    return 'Component {} in Product'.format(c)   # XMEAS 37–41, мол.%


# Базовый режим Downs & Vogel (1993) — для контрольного теста
BASE_CASE = {
    'Purge Rate': 0.33712, 'Stripper Underflow': 22.949, 'Stripper Temp': 65.731,
    'Stripper Steam Flow': 230.31, 'Compressor Work': 341.43,
    'Component A in Purge': 32.958, 'Component B in Purge': 13.823,
    'Component C in Purge': 23.978, 'Component D in Purge': 1.2565,
    'Component E in Purge': 18.579, 'Component F in Purge': 2.2633,
    'Component G in Purge': 4.8436, 'Component H in Purge': 2.2986,
    'Component D in Product': 0.01787, 'Component E in Product': 0.83570,
    'Component F in Product': 0.09858, 'Component G in Product': 53.724,
    'Component H in Product': 43.828,
}


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def to_dict(data, st_state_window=None):
    """dict / Series / DataFrame -> dict средних значений; снимает префикс 'xmeas_'.
    Для DataFrame усредняются последние st_state_window строк (или все)."""
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            if st_state_window and len(data) > st_state_window:
                data = data.iloc[-st_state_window:]
            data = data.mean(numeric_only=True)
        if isinstance(data, pd.Series):
            data = data.to_dict()
    except ImportError:
        pass
    out = {}
    for k, v in dict(data).items():
        key = k[6:] if isinstance(k, str) and k.startswith('xmeas_') else k
        out.setdefault(key, v)
    return out


def _get(d, key):
    if key not in d:
        raise KeyError('Нет переменной "{}" во входных данных'.format(key))
    v = float(d[key])
    if math.isnan(v):
        raise ValueError('Переменная "{}" = NaN'.format(key))
    return v


def product_molar_flow(d):
    """Мольный расход продукта, кгмоль/ч = 9,21 кгмоль/м3 × XMEAS 17 (Downs & Vogel)."""
    return PRODUCT_KGMOL_PER_M3 * _get(d, NAMES['product_rate'])


def purge_molar_flow(d):
    """Мольный расход продувки, кгмоль/ч = 44,79 × XMEAS 10 (Downs & Vogel)."""
    return PURGE_KGMOL_PER_KSCM * _get(d, NAMES['purge_rate'])


# ------------------------------------------------------------------
# 1. Стандартная функция затрат (Table 9)
# ------------------------------------------------------------------

def calc_tep_operating_cost(data, st_state_window=None):
    """Эксплуатационные затраты TEP по Downs & Vogel (1993), Table 9, $/ч.

    cost = purge_loss + product_loss + compressor_cost + steam_cost
      purge_loss   = 44,79·XMEAS10 · Σ_{A,C..H} (x_i/100)·C_i
      product_loss = 9,21·XMEAS17  · Σ_{D,E,F} (x_i/100)·C_i
      compressor   = 0,0536·XMEAS20
      steam        = 0,0318·XMEAS19

    Возвращает (total_cost, components: dict).
    """
    d = to_dict(data, st_state_window)

    purge_value = sum(_get(d, purge_key(c)) / 100.0 * COMPONENT_COST[c]
                      for c in PURGE_COMPONENTS)                        # $/кгмоль
    product_value = sum(_get(d, product_key(c)) / 100.0 * COMPONENT_COST[c]
                        for c in PRODUCT_LOSS_COMPONENTS)               # $/кгмоль

    comp = {
        'purge_value_per_kgmol': purge_value,
        'purge_molar_flow': purge_molar_flow(d),
        'product_loss_value_per_kgmol': product_value,
        'product_molar_flow': product_molar_flow(d),
    }
    comp['purge_loss'] = comp['purge_molar_flow'] * purge_value
    comp['product_loss'] = comp['product_molar_flow'] * product_value
    comp['compressor_cost'] = COMPRESSOR_COST * _get(d, NAMES['compressor'])
    comp['steam_cost'] = STEAM_COST * _get(d, NAMES['steam'])
    comp['total_cost'] = (comp['purge_loss'] + comp['product_loss']
                          + comp['compressor_cost'] + comp['steam_cost'])
    return comp['total_cost'], comp


# ------------------------------------------------------------------
# 2. Авторское расширение: прибыль
# ------------------------------------------------------------------

def calc_tep_profit_extended(data, price_G, price_H, st_state_window=None):
    """АВТОРСКОЕ РАСШИРЕНИЕ стандартной модели (не Table 9).

    profit = revenue − operating_cost,
    revenue = F_prod · (x_G·price_G + x_H·price_H),  F_prod = 9,21·XMEAS17.

    price_G, price_H ($/кгмоль) задаются явно: у них должен быть отдельный
    источник, либо они объявляются модельным допущением. Затраты на сырьё,
    утилизацию и некондицию сюда не входят.
    """
    d = to_dict(data, st_state_window)
    cost, comp = calc_tep_operating_cost(d)
    F = product_molar_flow(d)
    revenue = F * (_get(d, product_key('G')) / 100.0 * price_G
                   + _get(d, product_key('H')) / 100.0 * price_H)
    comp = dict(comp)
    comp['revenue'] = revenue
    comp['profit'] = revenue - cost
    return comp['profit'], comp
