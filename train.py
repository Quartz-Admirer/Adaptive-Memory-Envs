import torch
import os
from docopt import docopt

from trainer import PPOTrainer
from yaml_parser import YamlParser

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/endless_tmaze.yaml]
        --run-id=<path>            Specifies the base tag for saving models and logs. If not set, uses run_id from config.
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    config_path = options["--config"]
    cpu = options["--cpu"]
    
    # Parse the yaml config file.
    config = YamlParser(config_path).get_config()
    
    # Определяем ID запуска и количество ранов
    base_run_id = options["--run-id"] if options["--run-id"] else config["run_id"]
    num_runs = config.get("n_runs", 1)

    # Determine the device to be used for training
    device = torch.device("cpu") if cpu or not torch.cuda.is_available() else torch.device("cuda")
    if device.type == 'cuda':
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # --- Training Loop for Multiple Runs ---
    for i in range(num_runs):
        run_id = f"{base_run_id}_run_{i+1}"
        print(f"\n{'='*40}\n--- Starting Run {i+1}/{num_runs} with ID: {run_id} ---\n{'='*40}\n")
        
        # Initialize the PPO trainer and commence training
        trainer = PPOTrainer(config, run_id=run_id, device=device)
        trainer.run_training()
        trainer.close()
        
    print(f"\nAll {num_runs} training runs are complete.")
    print(f"Best models saved to './models/' with prefix 'best_model_{base_run_id}_...'.")
    print("Use validate_models.py for cross-validation.")

if __name__ == "__main__":
    main()
