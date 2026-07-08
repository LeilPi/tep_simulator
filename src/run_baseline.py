import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytep.siminterface import SimInterface
from src.simulator_config import DEFAULT_DURATION
from src.save_data import save_simulation_data
def run_baseline(duration=DEFAULT_DURATION, save_path="data/baseline/baseline_run_001.csv"):
    tep = SimInterface()
    tep.setup()
    print(f"Запуск базовой симуляции на {duration} часов...")
    tep.simulate(duration=duration)
    df = tep.process_data
    print(f"Получено {len(df)} строк данных.")
    save_simulation_data(df, save_path, disturbance_id=0)
    tep.reset()
    return df
if __name__ == "__main__":
    # Если скрипт запускается напрямую, можно передать параметры через аргументы командной строки
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DURATION
    save_path = sys.argv[2] if len(sys.argv) > 2 else "data/baseline/baseline_run_001.csv"
    run_baseline(duration, save_path)
