import numpy as np
import torch

from gymnasium import spaces

class Buffer():
    """The buffer stores and prepares the training data. It supports transformer-based memory policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, action_space_shape:tuple, max_episode_length:int, device:torch.device) -> None:
        """
        Arguments:
            config {dict} -- Полный конфигурационный словарь из YAML файла.
            observation_space {spaces.Box} -- The observation space of the agent
            action_space_shape {tuple} -- Shape of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
            device {torch.device} -- The device that will be used for training
        """
        # --- Адаптивная загрузка параметров ---
        # Проверяем, есть ли секция 'defaults' для параметров PPO
        if "defaults" in config:
            ppo_config = config["defaults"]
        else:
            # Если нет, считаем, что параметры находятся на верхнем уровне
            ppo_config = config

        # Извлекаем параметры PPO
        self.device = device
        self.n_workers = ppo_config["n_workers"]
        self.worker_steps = ppo_config["worker_steps"]
        self.n_mini_batches = ppo_config["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        self.max_episode_length = max_episode_length

        # Извлекаем параметры трансформера (они всегда на верхнем уровне)
        transformer_config = config["transformer"]
        self.memory_length = transformer_config["memory_length"]
        self.num_blocks = transformer_config["num_blocks"]
        self.embed_dim = transformer_config["embed_dim"]

        # --- Инициализация хранилищ буфера ---
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))
        
        self.memories = []
        self.memory_mask = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.bool)
        self.memory_index = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        self.memory_indices = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.long)

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary."""
        samples = {
            "actions": self.actions,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "obs": self.obs,
            "memory_mask": self.memory_mask,
            "memory_index": self.memory_index,
            "memory_indices": self.memory_indices,
        }
        
        if self.memories:
            self.memories = torch.stack(self.memories, dim=0)

        self.samples_flat = {}
        for key, value in samples.items():
            self.samples_flat[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    def mini_batch_generator(self):
        """A generator that returns a dictionary containing the data of a whole minibatch."""
        if self.batch_size == 0:
            return

        indices = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            if end > self.batch_size:
                continue # Пропускаем неполный батч
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            
            # Специальная обработка для памяти
            mem_indices_for_batch = self.samples_flat["memory_index"][mini_batch_indices]
            if len(self.memories) > 0:
                 mini_batch["memories"] = self.memories[mem_indices_for_batch]
            else:
                 # Создаем пустой тензор, если память не используется
                 mini_batch["memories"] = torch.empty(0)


            for key, value in self.samples_flat.items():
                if key != "memory_index":
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Generalized advantage estimation (GAE)"""
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not()
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]