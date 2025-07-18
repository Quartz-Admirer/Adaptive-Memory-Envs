# ШАГ 1: Все импорты находятся наверху
import argparse
import popgym
import ray
import torch
from torch import nn

# Импортируем модели из popgym
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_ffm import RayFFM
from popgym.baselines.ray_models.ray_shm import SHMAgent

# КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Импортируем PPOConfig под защищенным именем,
# чтобы popgym не мог его перезаписать.
from ray.rllib.algorithms.ppo import PPOConfig as RLLIB_PPOConfig

# --- Функции конфигурации (без импортов внутри) ---

def config_gru(args):
    bptt_size = 1024
    config = {
        "model": {
            "max_seq_len": bptt_size,
            "custom_model": GRU,
            "custom_model_config": {
                "preprocessor_input_size": 128,
                "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "hidden_size": 256,
                "postprocessor": nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
                "postprocessor_output_size": 64,
                "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64),
                "num_recurrent_layers": 1,
            },
        },
        "num_gpus": args.fgpu,
        "num_workers": 2,
        "sgd_minibatch_size": bptt_size * 8,
        "train_batch_size": bptt_size * 64,
        "env": f"popgym-{args.env}-v0",
        "rollout_fragment_length": bptt_size,
        "framework": "torch",
        "horizon": bptt_size,
        "batch_mode": "complete_episodes",
    }
    return config

def config_ffm(args):
    bptt_size = 1024
    config = {
        "model": {
            "max_seq_len": bptt_size,
            "custom_model": RayFFM,
            "custom_model_config": {
                "preprocessor_input_size": 128,
                "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "hidden_size": 64,
                "mem_size": args.m,
                "post_size": args.post_size,
                "postprocessor": nn.Sequential(nn.Linear(args.post_size, 64), nn.ReLU()),
                "postprocessor_output_size": 64,
                "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64),
                "num_recurrent_layers": 1,
            },
        },
        "num_gpus": args.fgpu,
        "num_workers": 2,
        "sgd_minibatch_size": bptt_size * 8,
        "train_batch_size": bptt_size * 64,
        "env": f"popgym-{args.env}-v0",
        "rollout_fragment_length": bptt_size,
        "framework": "torch",
        "horizon": bptt_size,
        "batch_mode": "complete_episodes",
    }
    return config

def config_shm(args):
    bptt_size = 1024
    config = {
        "model": {
            "max_seq_len": bptt_size,
            "custom_model": SHMAgent,
            "custom_model_config": {
                "preprocessor_input_size": 128,
                "preprocessor_output_size": 64,
                "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "preprocessor2": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
                "hidden_size": args.h,
                "embedding_size": 128,
                "mem_size": args.m,
                "post_size": args.post_size,
                "postprocessor": nn.Sequential(nn.Linear(args.post_size, 64), nn.ReLU()),
                "postprocessor_output_size": 64,
                "actor": nn.Linear(64, 64),
                "critic": nn.Linear(64, 64)
            },
        },
        "num_gpus": args.fgpu,
        "num_workers": 2,
        "sgd_minibatch_size": int(bptt_size * 8 * args.bscale),
        "train_batch_size": int(bptt_size * 64 * args.bscale),
        "env": f"popgym-{args.env}-v0",
        "rollout_fragment_length": bptt_size,
        "framework": "torch",
        "horizon": bptt_size,
        "batch_mode": "complete_episodes",
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

    if args.model == "gru":
        config_dict = config_gru(args)
    elif args.model == "ffm":
        config_dict = config_ffm(args)
    elif args.model == "shm":
        config_dict = config_shm(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Используем наше защищенное имя для создания объекта
    algo_config = (
        RLLIB_PPOConfig()
        .environment(env=config_dict["env"], horizon=config_dict["horizon"])
        .framework(config_dict["framework"])
        .env_runners(
            num_env_runners=config_dict["num_workers"],
            rollout_fragment_length=config_dict["rollout_fragment_length"],
            batch_mode=config_dict["batch_mode"],
        )
        .training(
            gamma=0.99,
            lr=5e-5,
            vf_loss_coeff=1.0,
            train_batch_size=config_dict["train_batch_size"],
            sgd_minibatch_size=config_dict["sgd_minibatch_size"],
        )
        .resources(num_gpus=config_dict["num_gpus"])
        .debugging(seed=42)
        .model(**config_dict["model"])
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    num_gpus_for_ray = 1 if args.gpu > 0 and torch.cuda.is_available() else 0
    ray.init(ignore_reinit_error=True, num_cpus=10, num_gpus=num_gpus_for_ray)

    ray.tune.run(
        "PPO",
        config=algo_config,
        num_samples=args.nrun,
        stop={"timesteps_total": 15_000_000},
        storage_path=f"/home/jovyan/persistent_volume/results/{args.env}/{args.model}/",
    )
