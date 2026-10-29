# Стандартная постановка TEP: минимизация эксплуатационных затрат

Целевая функция — эксплуатационные затраты по Downs & Vogel (1993), Table 9
(`src/tep_economics.py`). Производительность фиксирована (ProductionSP = 22,89),
варьируются 5 уставок. Старые скрипты (`main_function_1.py`, `run_cube_5.py` и др.)
не изменены.

| № | Команда | Что делает | Время |
|---|---|---|---|
| 1 | `python tests\test_tep_economics.py` | Тест функции затрат: 170,6 $/ч на базовом режиме D&V | секунды |
| 2 | `python src\recompute_operating_cost.py` | Пересчёт затрат для 75 старых опытов → `data\doe_extended\runs\results_operating_cost.csv` | секунды |
| 3 | `python src\generate_hypercube_cost.py` | Новый план, 47 точек → `data\doe_cost\design\` | секунды |
| 4 | `python src\run_cube_cost.py --max 2` | Пробный прогон 2 точек | ~6 мин |
| 5 | `python src\run_nominal_cost.py --repeats 3` | Номинал 3 раза: точка сравнения и шум | ~10 мин |
| 6 | `python src\run_cube_cost.py` | Весь план; после сбоя запустить снова — продолжит | ~2,5 ч |
| 7 | `python src\build_model_cost.py` | Модель затрат, минимум, график `figures\cost_model.png` | секунды |
| 8 | `python src\validate_cost_optimum.py --repeats 3` | Проверка минимума на симуляторе | ~10 мин |

Усреднение по участку 20–30 ч каждого прогона (вместо последних 0,5 ч).
Шум оценивается по 2-часовым блокам; улучшение меньше 2σ шума не засчитывается.

## Уточнение после полного прогона (v2)

Зачем: проверить, не «подогнан» ли оптимум якорными точками; измерить прямо на
симуляторе влияние клапана пара и мешалки; перепрогнать опыты, не вышедшие на стационар.
Первые результаты (`model_cost.pkl`, `validation_cost.csv`, `cost_model.png`) не затираются.

| № | Команда | Что делает | Время |
|---|---|---|---|
| 9 | `python src\robustness_cost.py` | Модель на 4 вариантах данных (все / без якорей / без неустановившихся / без обоих) | секунды |
| 10 | `python src\check_setpoint_effects.py` | Оптимум с паром 25 %, 50 % и мешалкой 250 — прямое влияние уставок | ~20 мин |
| 11 | `python src\rerun_unsteady.py` | Опыты с σ блоков > 2 × медианы → 50 ч, усреднение 40–50 ч | ~30 мин |
| 12 | `python src\merge_results_v2.py` | Сборка `results_cost_v2.csv` | секунды |
| 13 | `python src\robustness_cost.py --results data\doe_cost\runs\results_cost_v2.csv` | Повтор проверки 9 на уточнённых данных | секунды |
| 14 | `python src\build_model_cost.py --results data\doe_cost\runs\results_cost_v2.csv --tag v2` | Модель v2 → `model_cost_v2.pkl`, `cost_model_v2.png` | секунды |
| 15 | `python src\validate_cost_optimum.py --tag v2 --repeats 1` | Проверка оптимума v2 (pyTEP детерминирован — одного прогона достаточно) | ~6 мин |

Шаги 10 и 11 независимы, порядок любой; оба можно перезапускать после сбоя.
