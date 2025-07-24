import numpy as np
import torch

from torch import nn

# Убедитесь, что импорт правильный
from environments.endless_tmaze_env import EndlessTMazeGym

def create_env(config:dict, render:bool=False):
    """
    Инициализирует среду, передавая ей полный конфигурационный словарь.
    
    Аргументы:
        config {dict}: Словарь с параметрами для среды (num_corridors, lengths и т.д.)
        render {bool}: Флаг для режима рендеринга (пока не используется).

    Returns:
        {EndlessTMazeGym}: Возвращает экземпляр среды.
    """
    # Теперь мы не ищем 'reset_params', а передаем весь словарь 'config' напрямую.
    # Внутри EndlessTMazeGym уже сам разберет нужные ему поля.
    # Дополнительные параметры, такие как render_mode, можно добавить в словарь перед передачей, если нужно.
    env_params = config.copy()
    if render:
        env_params["render_mode"] = "human"
        
    return EndlessTMazeGym(env_config=env_params)

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    if current_step > max_decay_steps or initial == final:
        return final
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)
    
def batched_index_select(input_tensor, dim, index):
    """
    Selects values from the input tensor at the given indices along the given dimension.
    This function is similar to torch.index_select, but it supports batched indices.
    """
    # Создаем представление (view) индекса, которое будет совместимо с формой входного тензора
    for ii in range(1, len(input_tensor.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    
    # Расширяем индекс до размеров входного тензора для корректной работы gather
    expanse = list(input_tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    
    return torch.gather(input_tensor, dim, index)

def process_episode_info(episode_info:list) -> dict:
    """Extracts the mean and std of completed episode statistics like length and total reward.

    Arguments:
        episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

    Returns:
        {dict} -- Processed episode results (computes the mean and std for most available keys)
    """
    result = {}
    if len(episode_info) > 0:
        # Собираем все ключи из словарей, чтобы обработать все возможные метрики
        all_keys = set().union(*(d.keys() for d in episode_info))
        for key in all_keys:
            # Убеждаемся, что значение является числовым для вычисления статистики
            valid_values = [info[key] for info in episode_info if key in info and isinstance(info[key], (int, float))]
            if valid_values:
                result[key + "_mean"] = np.mean(valid_values)
                result[key + "_std"] = np.std(valid_values)
    return result

class Module(nn.Module):
    """nn.Module is extended by functions to compute the norm and the mean of this module's parameters."""
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm."""
        grads = []
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean."""
        grads = []
        for name, parameter in self.named_parameters():
             if parameter.grad is not None:
                grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None