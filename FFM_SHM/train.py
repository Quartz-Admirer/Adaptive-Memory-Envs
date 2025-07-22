import argparse
import os
import sys
import ray
import torch
from torch import nn
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO

# Импортируем наши компоненты
from config import get_config
from popgym.envs.endless_tmaze_env import EndlessTMazeGym
# Модели из popgym, как в вашем оригинальном файле
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_ffm import RayFFM
from popgym.baselines.ray_models.ray_shm import SHMAgent

def get_model_class(model_name):
    """Возвращает класс модели по названию."""
    if model_name == "gru":
        return GRU
    elif model_name == "ffm":
        return RayFFM
    elif model_name == "shm":
        return SHMAgent
    else:
        raise ValueError(f"Unknown model: {model_name}")

def build_model_config(cli_args):
    """
    Создает правильный `custom_model_config` для выбранной модели.
    Эта функция восстанавливает логику из вашего оригинального train.py.
    """
    model_name = cli_args.model
    bptt_size = 1024 # Константа из вашего конфига
    
    # --- Логика построения конфига для каждой модели ---
    if model_name == "gru":
        custom_model_config = {
            "preprocessor_input_size": 128, "preprocessor_output_size": 64,
            "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
            "hidden_size": 256,
            "postprocessor": nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
            "postprocessor_output_size": 64, "actor": nn.Linear(64, 64),
            "critic": nn.Linear(64, 64), "num_recurrent_layers": 1,
        }
    elif model_name == "ffm":
        custom_model_config = {
            "preprocessor_input_size": 128, "preprocessor_output_size": 64,
            "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
            "hidden_size": 64, "mem_size": cli_args.m, "post_size": cli_args.post_size,
            "postprocessor": nn.Sequential(nn.Linear(cli_args.post_size, 64), nn.ReLU()),
            "postprocessor_output_size": 64, "actor": nn.Linear(64, 64),
            "critic": nn.Linear(64, 64), "num_recurrent_layers": 1,
        }
    elif model_name == "shm":
        custom_model_config = {
            "preprocessor_input_size": 128, "preprocessor_output_size": 64,
            "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
            "preprocessor2": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
            "hidden_size": cli_args.h, "embedding_size": 128, "mem_size": cli_args.m,
            "post_size": cli_args.post_size,
            "postprocessor": nn.Sequential(nn.Linear(cli_args.post_size, 64), nn.ReLU()),
            "postprocessor_output_size": 64, "actor": nn.Linear(64, 64),
            "critic": nn.Linear(64, 64)
        }
    else:
        custom_model_config = {}

    return {
        "max_seq_len": bptt_size,
        "custom_model": get_model_class(model_name),
        "custom_model_config": custom_model_config,
    }

def train(cli_args, experiment_config):
    """Запускает обучение с использованием ray.tune."""
    print("--- Starting Training ---")
    
    bptt_size = 1024
    ppo_config = {
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
        "framework": "torch",
        "num_gpus": cli_args.fgpu,
        "num_workers": 4,
        "rollout_fragment_length": bptt_size,
        "train_batch_size": int(bptt_size * 64 * cli_args.bscale),
        "sgd_minibatch_size": int(bptt_size * 8 * cli_args.bscale),
        "horizon": bptt_size,
        "batch_mode": "truncate_episodes",
        "gamma": 0.99,
        "lr": 5e-5,
        "vf_loss_coeff": 1.0,
        "model": build_model_config(cli_args), # <--- ИСПРАВЛЕНИЕ ЗДЕСЬ
        "env": "EndlessTMazeGym-v0",
        "env_config": experiment_config["train_env"],
    }

    checkpoint_config = ray.tune.CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_score_order="max",
    )
    storage_path = os.path.abspath("./results/")

    analysis = ray.tune.run(
        "PPO",
        name=f"{cli_args.env}_{cli_args.model}",
        config=ppo_config,
        num_samples=cli_args.nrun,
        stop={"timesteps_total": experiment_config["total_timesteps"]},
        storage_path=storage_path,
        metric="episode_reward_mean",
        mode="max",
        checkpoint_config=checkpoint_config,
        log_to_file=True,
        verbose=1,
    )

    print("\n--- Training Finished ---")
    best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="all")
    if best_trial:
        best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
        if best_checkpoint:
            print(f"Best trial: {best_trial.trial_id}")
            print(f"Best checkpoint found at: {best_checkpoint.path}")
            return best_checkpoint.path
    return None

def validate(cli_args, experiment_config, checkpoint_path):
    """Запускает валидацию лучшей модели."""
    print("\n--- Running Validation ---")
    print(f"Loading model from checkpoint: {checkpoint_path}")

    ppo_config = {
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,
        "framework": "torch",
        "num_workers": 1,
        "env": "EndlessTMazeGym-v0",
        "env_config": experiment_config["validation"]["env"],
        "model": build_model_config(cli_args), # <--- Также добавляем сюда
    }
    
    agent = PPO(config=ppo_config)
    agent.restore(checkpoint_path)

    rewards, lengths, successes = [], [], []
    eval_env = EndlessTMazeGym(env_config=experiment_config["validation"]["env"])

    for _ in range(100):
        obs, _ = eval_env.reset()
        done, truncated = False, False
        episode_reward, episode_length = 0, 0
        state = agent.get_policy().get_initial_state()
        
        while not (done or truncated):
            action, state, _ = agent.compute_single_action(obs, state=state, explore=False)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        if "success" in info:
            successes.append(info["success"])

    print("\n--- Validation Results ---")
    print(f"Mean Reward: {np.mean(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.1f}")
    if successes:
        print(f"Success Rate: {np.mean(successes):.2%}")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(description='Popgym Benchmark with Validation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate'], help="Mode: 'train' or 'validate'")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to checkpoint file for validation.")
    parser.add_argument('--env', type=str, default='EndlessTMazeEnv', help="Environment name.")
    parser.add_argument('--model', type=str, default='gru', choices=['gru', 'ffm', 'shm'], help="Model architecture.")
    parser.add_argument('--nrun', type=int, default=3, help="Number of training runs.")
    parser.add_argument('--gpu', type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument('--bscale', type=float, default=1.0)
    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--m', type=int, default=72)
    parser.add_argument('--post_size', type=int, default=1024)
    
    args = parser.parse_args()
    args.fgpu = 1.0 / args.nrun if args.gpu > 0 and torch.cuda.is_available() else 0

    exp_config = get_config()
    register_env("EndlessTMazeGym-v0", lambda config: EndlessTMazeGym(env_config=config))
    
    num_gpus_for_ray = 1 if args.gpu > 0 and torch.cuda.is_available() else 0
    ray.init(ignore_reinit_error=True, num_cpus=10, num_gpus=num_gpus_for_ray)

    if args.mode == 'train':
        best_checkpoint_path = train(args, exp_config)
        if best_checkpoint_path:
            validate(args, exp_config, best_checkpoint_path)
        else:
            print("Could not find a best checkpoint to validate.")
            
    elif args.mode == 'validate':
        if not args.checkpoint_path:
            print("Error: --checkpoint_path must be provided for validation mode.")
            sys.exit(1)
        validate(args, exp_config, args.checkpoint_path)

if __name__ == '__main__':
    main()

