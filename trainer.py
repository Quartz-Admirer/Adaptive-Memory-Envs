import numpy as np
import os
import pickle
import time
import torch

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from buffer import Buffer
from model import ActorCriticModel
from utils import batched_index_select, create_env, polynomial_decay, process_episode_info
from worker import Worker

class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """
        Инициализирует все компоненты, адаптируясь к структуре YAML файла.
        """
        self.config = config
        self.device = device
        self.run_id = run_id
        
        # --- Адаптивная загрузка конфигурации ---
        if "defaults" in config:
            self.ppo_config = config["defaults"]
        else:
            self.ppo_config = config
        self.train_env_config = config["train_env"]

        # --- Извлечение параметров ---
        self.num_workers = self.ppo_config["n_workers"]
        self.worker_steps = self.ppo_config["worker_steps"]
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]
        
        self.transformer_config = config["transformer"]
        self.memory_length = self.transformer_config["memory_length"]
        self.num_blocks = self.transformer_config["num_blocks"]
        self.embed_dim = self.transformer_config["embed_dim"]
        
        self.validation_config = config.get("validation", {"enabled": False})

        # --- Настройка логгирования ---
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # --- Инициализация компонентов ---
        print("Step 1: Init dummy environment for spaces")
        dummy_env = create_env(self.train_env_config)
        observation_space = dummy_env.observation_space
        self.action_space_shape = (dummy_env.action_space.n,)
        self.max_episode_length = dummy_env.max_episode_steps
        dummy_env.close()

        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, observation_space, self.action_space_shape, self.max_episode_length, self.device)

        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(config, observation_space, self.action_space_shape, self.max_episode_length).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        print("Step 4: Init training workers")
        self.workers = [Worker(self.train_env_config) for _ in range(self.num_workers)]
        self.obs = np.zeros((self.num_workers,) + observation_space.shape, dtype=np.float32)
        for w, worker in enumerate(self.workers):
            worker.child.send(("reset", None))
            self.obs[w], _ = worker.child.recv()

        # --- Управление памятью GTrXL ---
        self.worker_ids = range(self.num_workers)
        self.worker_current_episode_step = torch.zeros((self.num_workers, ), dtype=torch.long)
        self.memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        self.memory_indices = self._create_memory_indices()

        # --- Логика для чекпоинтинга лучшей модели ---
        if self.validation_config.get("enabled", False):
            self.best_metric_value = -np.inf
            self.metric_to_track = self.validation_config.get("best_model_metric", "reward_mean")
            self.best_model_path = f"./models/best_model_{self.run_id}.nn"
            print(f"Checkpointing enabled. Tracking metric: '{self.metric_to_track}'. Best model will be saved to {self.best_model_path}")

    def _create_memory_indices(self) -> torch.Tensor:
        """Создает тензор для индексации скользящего окна памяти."""
        if self.max_episode_length <= self.memory_length:
            return torch.arange(0, self.memory_length).unsqueeze(0).repeat(self.max_episode_length, 1).long()
            
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim=0).long()
        indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        return torch.cat((repetitions, indices))

    def run_training(self) -> None:
        """Основной цикл обучения агента."""
        print(f"Step 5: Starting training using {self.device}")
        episode_infos = deque(maxlen=100)

        for update in range(self.ppo_config["updates"]):
            if self.validation_config.get("enabled", False) and update > 0 and update % self.validation_config.get("val_every", 50) == 0:
                val_metrics = self.run_validation(model=self.model, update_step=update)
                
                # Логика сохранения лучшей модели
                current_metric = val_metrics.get(self.metric_to_track)
                if current_metric is not None and current_metric > self.best_metric_value:
                    self.best_metric_value = current_metric
                    print(f"\nNew best model found! {self.metric_to_track}: {current_metric:.3f}. Saving model to {self.best_model_path}...\n")
                    self._save_model(path=self.best_model_path)

            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            sampled_episode_info = self._sample_training_data()
            self.buffer.prepare_batch_dict()

            training_stats, grad_info = self._train_epochs(learning_rate, clip_range, beta)
            if training_stats:
                training_stats = np.mean(training_stats, axis=0)
            
            episode_infos.extend(sampled_episode_info)
            processed_info = process_episode_info(episode_infos)
            self._log_stats(update, training_stats, processed_info, grad_info)

        print("Training complete. Best model saved via checkpointing.")

    def run_validation(self, model_path: str = None, model: ActorCriticModel = None, update_step: int = 0) -> dict:
        """
        Запускает валидацию и возвращает словарь с посчитанными метриками.
        """
        print("\n--- Running Validation ---")
        
        val_env_config = self.validation_config.get("env")
        if not val_env_config:
            print("Warning: 'validation.env' configuration not found. Skipping validation.")
            return {}

        n_eval_episodes = self.validation_config.get("n_eval_episodes", 100)

        if model is None:
            if model_path and os.path.exists(model_path):
                state_dict, config = pickle.load(open(model_path, "rb"))
                dummy_env = create_env(config["train_env"])
                model = ActorCriticModel(config, dummy_env.observation_space, (dummy_env.action_space.n,), dummy_env.max_episode_steps).to(self.device)
                model.load_state_dict(state_dict)
                dummy_env.close()
                print(f"Loaded model from {model_path} for validation.")
            else:
                print(f"Warning: Model file not found at {model_path}. Skipping validation.")
                return {}
        
        model.eval()
        val_env = create_env(val_env_config)
        
        episode_rewards, episode_lengths, episode_successes = [], [], []

        for _ in range(n_eval_episodes):
            obs, _ = val_env.reset()
            terminated, truncated = False, False
            
            memory = torch.zeros((1, self.max_episode_length, self.num_blocks, self.embed_dim), device=self.device)
            current_step = 0

            while not (terminated or truncated):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs).unsqueeze(0).to(self.device)
                    mem_indices_step = self.memory_indices[current_step].unsqueeze(0).to(self.device)
                    mem_mask_step = self.memory_mask[torch.clip(torch.tensor([current_step]), 0, self.memory_length - 1)].to(self.device)
                    sliced_memory = batched_index_select(memory, 1, mem_indices_step)

                    policy, _, new_mem_item = model(obs_tensor, sliced_memory, mem_mask_step, mem_indices_step)
                    
                    memory[0, current_step] = new_mem_item.squeeze(0)
                    action = torch.stack([b.sample() for b in policy], dim=1).cpu().numpy()[0]
                
                obs, _, terminated, truncated, info = val_env.step(action)
                current_step += 1
            
            if info:
                episode_rewards.append(info.get("reward", 0))
                episode_lengths.append(info.get("length", 0))
                episode_successes.append(info.get("success", 0))

        val_env.close()
        
        results = {}
        if episode_rewards:
            results = {
                "reward_mean": np.mean(episode_rewards),
                "length_mean": np.mean(episode_lengths),
                "success_mean": np.mean(episode_successes)
            }
            
            if self.writer is not None:
                self.writer.add_scalar("validation/reward_mean", results["reward_mean"], update_step)
                self.writer.add_scalar("validation/length_mean", results["length_mean"], update_step)
                self.writer.add_scalar("validation/success_mean", results["success_mean"], update_step)
            
            print(f"Validation @ step {update_step}: Reward={results['reward_mean']:.2f}, Length={results['length_mean']:.1f}, Success={results['success_mean']:.2f}")

        model.train()
        print("--- Validation Complete ---\n")
        return results

    def _sample_training_data(self) -> list:
        episode_infos = []
        
        self.buffer.memories = [self.memory[w] for w in range(self.num_workers)]
        for w in range(self.num_workers):
            self.buffer.memory_index[w] = w

        for t in range(self.worker_steps):
            with torch.no_grad():
                self.buffer.obs[:, t] = torch.from_numpy(self.obs)
                self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
                self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]
                sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])
                
                policy, value, memory_item = self.model(torch.from_numpy(self.obs).to(self.device), sliced_memory, self.buffer.memory_mask[:, t], self.buffer.memory_indices[:,t])
                
                self.memory[self.worker_ids, self.worker_current_episode_step] = memory_item

                actions = [b.sample() for b in policy]
                log_probs = [b.log_prob(a) for b, a in zip(policy, actions)]
                
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
                self.buffer.values[:, t] = value

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            for w, worker in enumerate(self.workers):
                obs_new, reward, terminated, truncated, info = worker.child.recv()
                
                self.buffer.rewards[w, t] = reward
                self.buffer.dones[w, t] = terminated

                if terminated or truncated:
                    self.worker_current_episode_step[w] = 0
                    if info:
                        episode_infos.append(info)
                    
                    worker.child.send(("reset", None))
                    obs_new, _ = worker.child.recv()
                    
                    mem_index = int(self.buffer.memory_index[w, t].item())
                    if mem_index < len(self.buffer.memories):
                        self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()
                    
                    self.memory[w].zero_()
                    
                    if t < self.worker_steps - 1:
                        self.buffer.memories.append(self.memory[w])
                        self.buffer.memory_index[w, t + 1:] = len(self.buffer.memories) - 1
                else:
                    self.worker_current_episode_step[w] += 1
                
                self.obs[w] = np.asarray(obs_new, dtype=np.float32)

        last_value = self.get_last_value()
        self.buffer.calc_advantages(last_value, self.ppo_config["gamma"], self.ppo_config["lamda"])

        return episode_infos

    def get_last_value(self) -> torch.Tensor:
        """
        Корректно вычисляет ценность последнего состояния, используя
        правильно нарезанное окно памяти.
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(self.obs).to(self.device)
            indices_tensor = self.memory_indices[self.worker_current_episode_step]
            sliced_memory = batched_index_select(self.memory, 1, indices_tensor)
            mask_tensor = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
            _, last_value, _ = self.model(obs_tensor, sliced_memory, mask_tensor, indices_tensor)
        return last_value

    def _train_epochs(self, learning_rate: float, clip_range: float, beta: float) -> tuple:
        train_info, grad_info = [], {}
        if self.buffer.batch_size == 0:
            return train_info, grad_info
        
        for _ in range(self.ppo_config["epochs"]):
            mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
                for key, value in self.model.get_grad_norm().items():
                    if value is not None:
                        grad_info.setdefault(key, []).append(value)
        return train_info, grad_info

    def _train_mini_batch(self, samples: dict, learning_rate: float, clip_range: float, beta: float) -> list:
        memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
        policy, value, _ = self.model(samples["obs"], memory, samples["memory_mask"], samples["memory_indices"])

        log_probs, entropies = [], []
        for i, branch in enumerate(policy):
            log_probs.append(branch.log_prob(samples["actions"][:, i]))
            entropies.append(branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1)

        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2).mean()

        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2).mean()

        entropy_bonus = entropies.mean()

        loss = policy_loss + self.ppo_config["value_loss_coefficient"] * vf_loss - beta * entropy_bonus

        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.ppo_config["max_grad_norm"])
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]
        
    def _log_stats(self, update, training_stats, episode_result, grad_info):
        """Логирует статистику в консоль и Tensorboard."""
        if training_stats is not None and len(training_stats) > 0 and "reward_mean" in episode_result:
            result_str = (f"Update: {update:4} | "
                          f"Reward: {episode_result['reward_mean']:.2f} | "
                          f"Length: {episode_result['length_mean']:.1f} | "
                          f"Success: {episode_result.get('success_mean', 0):.2f} | "
                          f"Pi_Loss: {training_stats[0]:.3f} | V_Loss: {training_stats[1]:.3f} | "
                          f"Entropy: {training_stats[3]:.3f}")
            print(result_str)

        self._write_training_summary(update, training_stats, episode_result)
        self._write_gradient_summary(update, grad_info)

    def _write_training_summary(self, update, training_stats, episode_result):
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar(f"episode/{key}", episode_result[key], update)
        
        if training_stats is not None and len(training_stats) > 0:
            self.writer.add_scalar("losses/total_loss", training_stats[2], update)
            self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
            self.writer.add_scalar("losses/value_loss", training_stats[1], update)
            self.writer.add_scalar("losses/entropy", training_stats[3], update)
            self.writer.add_scalar("training/kl_divergence", training_stats[4], update)
            self.writer.add_scalar("training/clip_fraction", training_stats[5], update)
            self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
            self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)
        
    def _write_gradient_summary(self, update, grad_info):
        for key, value in grad_info.items():
            if value:
                self.writer.add_scalar(f"gradients/{key}", np.mean(value), update)

    def _save_model(self, path: str):
        """Сохраняет модель по указанному пути."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        
        pickle.dump((self.model.state_dict(), self.config), open(path, "wb"))
        print(f"Model saved to {path}")

    def close(self):
        """Закрывает воркеры и SummaryWriter."""
        print("Closing trainer and workers...")
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except Exception as e:
            print(f"Exception while closing workers: {e}")
        
        if self.writer is not None:
            self.writer.close()
        time.sleep(1.0)
