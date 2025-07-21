# config.py
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """
    Возвращает полную конфигурацию для эксперимента с ray.tune.
    """
    config = {
        # --- Общие параметры для запуска ---
        "run_id_prefix": "ray_ppo_run",
        "total_timesteps": 3_000_000,
        
        # --- Конфигурации для обучения и валидации ---
        "train_env": {
            "num_corridors": 5,
            "penalty": -0.01,
            "goal_reward": 1.0,
            "lengths": {
                "mode": "uniform",
                "max": 10,
                "min": 1,
            }
        },

        "validation": {
            "enabled": True, # Этот флаг пока не используется, но оставлен для будущих расширений
            "env": {
                "num_corridors": 5,
                "penalty": -0.01,
                "goal_reward": 1.0,
                "lengths": {
                    "mode": "fixed",
                    "max": 10,
                }
            }
        }
    }
    return config
