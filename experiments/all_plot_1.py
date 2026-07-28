import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from pathlib import Path


def plot_all_variables(baseline_path, disturbance_path, onset_time=None, output_dir="figures"):

    # Загружаем данные
    df_base = pd.read_csv(baseline_path)
    df_dist = pd.read_csv(disturbance_path)

    # Создаём папку для графиков, если её нет
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Определяем список переменных (все колонки, кроме time и disturbance_id)
    exclude_cols = ['time', 'disturbance_id']
    variables = [col for col in df_base.columns if col not in exclude_cols]

    print(f"Найдено {len(variables)} переменных для построения.")

    # Проходим по каждой переменной и строим график
    for var in variables:
        # СОЗДАЁМ ФИГУРУ И ОСИ ЯВНО (fig, ax)
        fig, ax = plt.subplots(figsize=(36, 12))

        # Строим линии
        ax.plot(df_base['time'], df_base[var], label='Базовый', color='#1f77b4', linewidth=2)
        ax.plot(df_dist['time'], df_dist[var], label='С возмущением', color='#d62728', linewidth=2, linestyle='--')

        if onset_time is not None:
            ax.axvline(x=onset_time, color='green', linestyle=':', linewidth=2, label='Включение возмущения')

        # Подписи
        ax.set_xlabel('Время, часы', fontsize=12)
        ax.set_ylabel(var, fontsize=12)
        ax.set_title(f'Сравнение переменной: {var}', fontsize=14)
        ax.legend(fontsize=10)

        # ===== НАСТРОЙКА СЕТКИ (теперь ax определён) =====
        # Основные деления каждые 2 часа
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # Вспомогательные деления каждые 0.5 часа
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        # Включаем сетку
        ax.grid(True, which='major', linestyle='-', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.7, color='lightgray', alpha=0.5)
        # =================================================

        # Ограничиваем ось X (по данным)
        ax.set_xlim(0, df_base['time'].max())

        # Сохраняем
        safe_name = var.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        filename = f"{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # закрываем фигуру, чтобы не занимала память

    print(f"Все графики сохранены в папку '{output_dir}'")


if __name__ == "__main__":
    plot_all_variables(
        baseline_path="data/baseline/baseline_run_001.csv",
        disturbance_path="data/disturbance/idv_4_run_001.csv",
        onset_time=5.0,
        output_dir="figures"
    )