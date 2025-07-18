import argparse
import popgym
import ray
import torch
from torch import nn
from ray.tune.registry import register_env

# ВАШИ ПРАВИЛЬНЫЕ ИМПОРТЫ
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_ffm import RayFFM
from popgym.baselines.ray_models.ray_shm import SHMAgent
from popgym.envs.endless_tmaze_env import EndlessTMazeGym

# --- Функции конфигурации ---
# ... (весь код для config_gru, config_ffm, config_shm остается здесь без изменений) ...
def config_gru(args):
    bptt_size = 1024
    config = {
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
        "model": {
            "max_seq_len": bptt_size, "custom_model": GRU,
            "custom_model_config": {
                "preprocessor_input_size": 128, "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "hidden_size": 256,
                "postprocessor": nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
                "postprocessor_output_size": 64, "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64), "num_recurrent_layers": 1,
            },
        },
        "num_gpus": args.fgpu, "num_workers": 2, "sgd_minibatch_size": bptt_size * 8,
        "train_batch_size": bptt_size * 64, "rollout_fragment_length": bptt_size,
        "framework": "torch", "horizon": bptt_size, "batch_mode": "complete_episodes",
        "gamma": 0.99, "lr": 5e-5, "vf_loss_coeff": 1.0,
    }
    return config

def config_ffm(args):
    bptt_size = 1024
    config = {
        "enable_rl_module_and_learner": False, "enable_env_runner_and_connector_v2": False,
        "model": {
            "max_seq_len": bptt_size, "custom_model": RayFFM,
            "custom_model_config": {
                "preprocessor_input_size": 128, "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "hidden_size": 64, "mem_size": args.m, "post_size": args.post_size,
                "postprocessor": nn.Sequential(nn.Linear(args.post_size, 64), nn.ReLU()),
                "postprocessor_output_size": 64, "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64), "num_recurrent_layers": 1,
            },
        },
        "num_gpus": args.fgpu, "num_workers": 2, "sgd_minibatch_size": bptt_size * 8,
        "train_batch_size": bptt_size * 64, "rollout_fragment_length": bptt_size,
        "framework": "torch", "horizon": bptt_size, "batch_mode": "complete_episodes",
        "gamma": 0.99, "lr": 5e-5, "vf_loss_coeff": 1.0,
    }
    return config

def config_shm(args):
    bptt_size = 1024
    config = {
        "enable_rl_module_and_learner": False, "enable_env_runner_and_connector_v2": False,
        "model": {
            "max_seq_len": bptt_size, "custom_model": SHMAgent,
            "custom_model_config": {
                "preprocessor_input_size": 128, "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "preprocessor2": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "hidden_size": args.h, "embedding_size": 128, "mem_size": args.m,
                "post_size": args.post_size,
                "postprocessor": nn.Sequential(nn.Linear(args.post_size, 64), nn.ReLU()),
                "postprocessor_output_size": 64, "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64)
            },
        },
        "num_gpus": args.fgpu, "num_workers": 2,
        "sgd_minibatch_size": int(bptt_size * 8 * args.bscale),
        "train_batch_size": int(bptt_size * 64 * args.bscale),
        "rollout_fragment_length": bptt_size, "framework": "torch", "horizon": bptt_size,
        "batch_mode": "complete_episodes", "gamma": 0.99, "lr": 5e-5, "vf_loss_coeff": 1.0,
    }
    return config

# --- Основной блок ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Popgym Benchmark')
    parser.add_argument('--env', type=str, default='AutoencodeEasy')
    parser.add_argument('--model', type=str, default='gru')
    parser.add_argument('--nrun', type=int, default=3)
    parser.add_argument('--bscale', type=float, default=1.0)
    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--m', type=int, default=72)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--post_size', type=int, default=1024)
    args = parser.parse_args()
    args.fgpu = 1.0 / args.nrun if args.gpu > 0 else 0

    # ===== ГЛАВНОЕ ИСПРАВЛЕНИЕ: ИЗМЕНЯЕМ LAMBDA-ФУНКЦИЮ =====
    # Мы игнорируем проблемный 'config' от Ray и создаем среду с параметрами по умолчанию.
    register_env("EndlessTMazeGym-v0", lambda config: EndlessTMazeGym())

    # Создаем базовый словарь конфигурации
    if args.model == "gru":
        config_dict = config_gru(args)
    elif args.model == "ffm":
        config_dict = config_ffm(args)
    elif args.model == "shm":
        config_dict = config_shm(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Устанавливаем правильное имя среды
    if args.env == "EndlessTMazeEnv":
        config_dict["env"] = "EndlessTMazeGym-v0"
    else:
        config_dict["env"] = f"popgym-{args.env}-v0"

    num_gpus_for_ray = 1 if args.gpu > 0 and torch.cuda.is_available() else 0
    ray.init(ignore_reinit_error=True, num_cpus=10, num_gpus=num_gpus_for_ray)

    ray.tune.run(
        "PPO",
        config=config_dict,
        num_samples=args.nrun,
        stop={"timesteps_total": 15_000_000},
        storage_path=f"/home/jovyan/persistent_volume/results/{args.env}/{args.model}/",
    )
