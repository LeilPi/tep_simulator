import sys
from pathlib import Path
import numpy as np
# Добавляем корневую папку, что бы импортировать директории
main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))
import pandas as pd
import numpy as np
def calc_tep_cost(data, st_satate_window=20):
    """
           Рассчитывает экономическую стоимость TEP.
           Параметры:
           data - pandas DataFrame с данными (xmeas, xmv, time, disturbance_id)
                  или pandas Series с одной строкой (текущее состояние)
           steady_state_window - количество последних точек для усреднения (если передана история)
    """
    # Если передан DataFrame с историей, берём последние N строк
    if isinstance(data, pd.DataFrame):
        if len(data) > st_satate_window:
            data = data.iloc[-st_satate_window:].mean()
        else:
            data.mean()
    if isinstance(data, pd.Series):
        data_dict = data.to_dict()
    else:
        data_dict = data
    # 1. Потери с продувкой (Purge Loss)
    # Состав компонентов в продувке
    purge_components = [
        'Component A in Purge',
        'Component B in Purge',
        'Component C in Purge',
        'Component D in Purge',
        'Component E in Purge',
        'Component F in Purge',
        'Component G in Purge',
        'Component H in Purge'
    ]
    purge_loss = 0.0
    for comp in purge_components:
        if comp in data_dict:
            purge_loss += data_dict[comp]

    purge_loss *= 0.1
    # 2. Потери с продуктом (Product Loss)
    # Компонент в продукте
    product_components = [
        'Component D in Product',
        'Component E in Product',
        'Component F in Product',
        'Component G in Product',
        'Component H in Product'
    ]
    product_loss = 0.0
    for comp in product_components:
        if comp in data_dict:
            product_loss += data_dict[comp]

    product_loss *= 0.5
    if 'Compressor Work' in data_dict:
        compressor_cost = 0.1 * data_dict['Compressor Work']

    stripper_cost = 0.0
    if 'Stripper Steam Flow' in data_dict:
        stripper_cost = 0.1 * data_dict['Stripper Steam Flow']

    total_cost = purge_loss + product_loss + compressor_cost + stripper_cost

    components = {
        'purge_loss': purge_loss,
        'product_loss': product_loss,
        'compressor_cost': compressor_cost,
        'stripper_cost': stripper_cost,
        'total_cost': total_cost
    }

    return total_cost, components

def calc_cost_from_files(xmeas_path, xmv_path, steady_state_window=20):
   """
    xmeas_path - путь к xmeas CSV
    xmv_path - путь к xmv CSV
    steady_state_window - окно усреднения
   """
    # Загружаем данные
   df_xmeas = pd.read_csv(xmeas_path)
   df_xmv = pd.read_csv(xmv_path)
   # Объединяем по времени
   df_combined = pd.merge(df_xmeas, df_xmv, on='time', how='outer')
   # Объединяем disturbance_id
   if 'disturbance_id_x' in df_combined.columns and 'disturbance_id_y' in df_combined.columns:
       df_combined['disturbance_id'] = df_combined['disturbance_id_x'].fillna(0) + df_combined['disturbance_id_y'].fillna(0)
       df_combined = df_combined.drop(columns=['disturbance_id_x', 'disturbance_id_y'])
   # Рассчитываем стоимость
   total_cost, components = calc_tep_cost(df_combined, steady_state_window)
   return total_cost, components, df_combined

def cost_for_steady_state(xmv_values, xmeas_steady):
# Объединяем xmv и xmeas в один словарь
    data = {**xmeas_steady, **xmv_values}
    total_cost, components = calc_tep_cost(data)
    return total_cost, components

if __name__ == "__main__":
    # --- Пример использования ---

    # Пути к вашим данным
    baseline_xmeas = "data/baseline/xmeas/baseline_xmeas.csv"
    baseline_xmv = "data/baseline/xmv/baseline_xmv.csv"

    print("=" * 60)
    print("Расчёт целевой функции для базового сценария")
    print("=" * 60)

    # 1. Рассчитываем стоимость из файлов
    total_cost, components, df = calc_cost_from_files(baseline_xmeas, baseline_xmv)

    print(f"\n📊 Компоненты стоимости:")
    for key, value in components.items():
        print(f"   {key}: {value:.4f}")
    print(f"\n💰 Итоговая стоимость: {total_cost:.4f}")

    # 2. Информация о данных
    print(f"\n📈 Данные: {len(df)} строк, {len(df.columns)} колонок")
    print(f"   Временной диапазон: {df['time'].min():.2f} - {df['time'].max():.2f} ч")

    # 3. Стоимость по времени (для визуализации)
    df['cost'] = df.apply(lambda row: calc_tep_cost(row)[0], axis=1)
    print(f"\n   Средняя стоимость за всё время: {df['cost'].mean():.4f}")
    print(f"   Стандартное отклонение: {df['cost'].std():.4f}")





    
























