import os
import subprocess
import copy
from ruamel.yaml import YAML


EXPERIMENT_DEFINITIONS = [
    {"id": 1, "mode": "fixed", "max_length": 5, "num_corridors": 1},
    {"id": 2, "mode": "fixed", "max_length": 5, "num_corridors": 3},
    {"id": 3, "mode": "fixed", "max_length": 5, "num_corridors": 5},
    {"id": 4, "mode": "fixed", "max_length": 5, "num_corridors": 10},
    {"id": 5, "mode": "fixed", "max_length": 10, "num_corridors": 1},
    {"id": 6, "mode": "fixed", "max_length": 10, "num_corridors": 3},
    {"id": 7, "mode": "fixed", "max_length": 10, "num_corridors": 5},
    {"id": 8, "mode": "fixed", "max_length": 10, "num_corridors": 10},
    {"id": 9, "mode": "uniform", "max_length": 5, "num_corridors": 1},
    {"id": 10, "mode": "uniform", "max_length": 5, "num_corridors": 3},
    {"id": 11, "mode": "uniform", "max_length": 5, "num_corridors": 5},
    {"id": 12, "mode": "uniform", "max_length": 5, "num_corridors": 10},
    {"id": 13, "mode": "uniform", "max_length": 10, "num_corridors": 1},
    {"id": 14, "mode": "uniform", "max_length": 10, "num_corridors": 3},
    {"id": 15, "mode": "uniform", "max_length": 10, "num_corridors": 5},
    {"id": 16, "mode": "uniform", "max_length": 10, "num_corridors": 10},
]

EXPERIMENTS = []
for definition in EXPERIMENT_DEFINITIONS:
    run_id = f"gtrl_len{definition['max_length']}_cor{definition['num_corridors']}_{definition['mode']}"
    
    lengths_config = {"mode": definition["mode"], "max": definition["max_length"]}
    if definition["mode"] == "uniform":
        lengths_config["min"] = 1

    config_override = {
        "train_env": {
            "num_corridors": definition["num_corridors"],
            "lengths": lengths_config
        }
    }
    
    EXPERIMENTS.append({"run_id": run_id, "config_override": config_override})


BASE_CONFIG_PATH = "./configs/endless_tmaze.yaml"

TEMP_CONFIG_PATH = "./configs/temp_experiment_config.yaml"

def update_dict(d, u):
    """Рекурсивно обновляет вложенный словарь."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def run():
    """
    Главная функция для запуска серии экспериментов.
    """
    yaml = YAML()
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            base_config = yaml.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Базовый файл конфигурации не найден по пути: {BASE_CONFIG_PATH}")
        return

    for i, experiment in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"ЗАПУСК ЭКСПЕРИМЕНТА {i+1}/{len(EXPERIMENTS)}: {experiment['run_id']}")
        print(f"{'='*60}\n")
        
        experiment_config = copy.deepcopy(base_config)
        update_dict(experiment_config, experiment["config_override"])

        if 'validation' in experiment_config and 'env' in experiment_config['validation']:
            experiment_config['validation']['env'] = experiment_config['train_env']

        experiment_config['run_id'] = experiment['run_id']

        try:
            with open(TEMP_CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(experiment_config, f)
            
            command = [
                "python",
                "train.py",
                "--config", TEMP_CONFIG_PATH
            ]
            
            subprocess.run(command, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"\nОШИБКА во время выполнения эксперимента '{experiment['run_id']}': {e}")
            print("Переход к следующему эксперименту...")
            continue
        finally:
            if os.path.exists(TEMP_CONFIG_PATH):
                os.remove(TEMP_CONFIG_PATH)
                
    print(f"\n{'='*60}")
    print("Все эксперименты завершены!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
