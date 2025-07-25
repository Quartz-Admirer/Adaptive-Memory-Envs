import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from model import ActorCriticModel
from utils import create_env, batched_index_select

# --- Конфигурация сценариев для кросс-валидации ---
# Этот список НЕ зависит от YAML конфига. Здесь вы определяете все среды для тестирования.
VALIDATION_ENV_CONFIGS = [
    {"id": 1, "mode": "fixed", "max_length": 5, "num_corridors": 3},
    {"id": 2, "mode": "fixed", "max_length": 10, "num_corridors": 3},
    {"id": 3, "mode": "fixed", "max_length": 15, "num_corridors": 3},
    {"id": 4, "mode": "uniform", "max_length": 5, "num_corridors": 3},
    {"id": 5, "mode": "uniform", "max_length": 10, "num_corridors": 3},
    {"id": 6, "mode": "uniform", "max_length": 15, "num_corridors": 3},
    {"id": 7, "mode": "fixed", "max_length": 10, "num_corridors": 1},
    {"id": 8, "mode": "fixed", "max_length": 10, "num_corridors": 5},
    {"id": 9, "mode": "fixed", "max_length": 10, "num_corridors": 10},
]

def evaluate_model(model: ActorCriticModel, model_config: dict, env_config: dict, device: torch.device):
    """
    Оценивает одну модель на одной среде.
    """
    model.eval()
    val_env = create_env(env_config)

    # Параметры из конфига модели
    transformer_conf = model_config["transformer"]
    memory_length = transformer_conf["memory_length"]
    num_blocks = transformer_conf["num_blocks"]
    embed_dim = transformer_conf["embed_dim"]
    max_episode_length = val_env.max_episode_steps

    # Создание маски и индексов памяти (аналогично trainer.py)
    memory_mask = torch.tril(torch.ones((memory_length, memory_length)), diagonal=-1).to(device)
    mem_indices_full = torch.arange(0, max_episode_length).long()

    rewards, successes = [], []
    n_eval_episodes = 100

    for _ in range(n_eval_episodes):
        obs, _ = val_env.reset()
        memory = torch.zeros((1, max_episode_length, num_blocks, embed_dim), device=device)
        terminated, truncated = False, False
        current_step = 0

        while not (terminated or truncated):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
                
                # Логика работы с памятью GTrXL
                mem_indices_step = mem_indices_full[max(0, current_step - memory_length + 1) : current_step + 1].unsqueeze(0)
                mem_mask_step = memory_mask[min(current_step, memory_length - 1)].unsqueeze(0)
                sliced_memory = batched_index_select(memory, 1, mem_indices_step)

                policy, _, new_mem_item = model(obs_tensor, sliced_memory, mem_mask_step, mem_indices_step)
                memory[0, current_step] = new_mem_item.squeeze(0)
                action = torch.stack([b.sample() for b in policy], dim=1).cpu().numpy()[0]
            
            obs, _, terminated, truncated, info = val_env.step(action)
            current_step += 1
        
        if info:
            rewards.append(info.get("reward", 0))
            successes.append(info.get("success", 0))

    val_env.close()
    return np.mean(successes) if successes else 0.0

def run_cross_validation():
    """Главная функция для запуска валидации."""
    parser = argparse.ArgumentParser(description="Запуск кросс-валидации для обученных моделей GTrXL.")
    parser.add_argument("--name", type=str, required=True, help="Базовое имя (run_id) для поиска моделей.")
    args = parser.parse_args()
    model_base_name = args.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Поиск всех лучших чекпоинтов для указанного имени
    model_dir = "./models"
    search_pattern = os.path.join(model_dir, f"best_model_{model_base_name}_run_*.nn")
    model_paths = sorted(glob.glob(search_pattern))

    if not model_paths:
        print(f"Ошибка: Не найдено ни одной модели по шаблону '{search_pattern}'")
        return

    print(f"Найдено {len(model_paths)} моделей для валидации:")
    for path in model_paths:
        print(f"- {os.path.basename(path)}")

    # Подготовка DataFrame для результатов
    model_names_short = [os.path.basename(p) for p in model_paths]
    val_scenarios_ids = [f"val_scenario_{c['id']}" for c in VALIDATION_ENV_CONFIGS]
    results_df = pd.DataFrame(index=model_names_short, columns=val_scenarios_ids, dtype=float)

    # Основной цикл: модель -> сценарий валидации
    for model_path in tqdm(model_paths, desc="Модели"):
        # Загрузка модели и ее конфига
        state_dict, model_config = pickle.load(open(model_path, "rb"))
        dummy_env = create_env(model_config["train_env"])
        model = ActorCriticModel(model_config, dummy_env.observation_space, (dummy_env.action_space.n,), dummy_env.max_episode_steps).to(device)
        model.load_state_dict(state_dict)
        dummy_env.close()

        for val_params in tqdm(VALIDATION_ENV_CONFIGS, desc="Сценарии", leave=False):
            # Формирование конфига для текущей среды валидации
            env_config = {
                "num_corridors": val_params["num_corridors"],
                "penalty": -0.01,
                "goal_reward": 1.0,
                "lengths": {
                    "mode": val_params["mode"],
                    "max": val_params["max_length"],
                    "min": 1 if val_params["mode"] == "fixed" else val_params["max_length"] // 2,
                }
            }
            
            success_rate = evaluate_model(model, model_config, env_config, device)
            
            col_id = f"val_scenario_{val_params['id']}"
            row_id = os.path.basename(model_path)
            results_df.loc[row_id, col_id] = success_rate

    # Вывод и сохранение результатов
    print("\n--- Результаты кросс-валидации (Success Rate) ---")
    print(results_df.round(3))

    output_filename = f"cross_validation_results_{model_base_name}.csv"
    results_df.to_csv(output_filename)
    print(f"\nТаблица с результатами сохранена в файл: {output_filename}")


if __name__ == "__main__":
    run_cross_validation()
