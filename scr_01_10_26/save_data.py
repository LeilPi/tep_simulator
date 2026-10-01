import pandas as pd
from pathlib import Path
def save_simulation_data(df, file_path, disturbance_id=0):
    #df – данные симуляции (это будет tep.process_data). file_path – путь, куда сохранять. disturbance_id – по умолчанию 0 (нет возмущения).
    df = df.copy()
    if 'disturbance_id' not in df.columns:
        df['disturbance_id'] = disturbance_id
        print(f"Добавлен столбец disturbance_id со значением {disturbance_id}")
    else:
        print("Столбец disturbance_id уже существует, сохранение без изменений.")
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    #Path(file_path) – создаёт объект пути. .parent – получает родительскую папку
    df.to_csv(file_path, index=False)
    print(f"Данные сохранены в {file_path} (строк: {len(df)})")
    