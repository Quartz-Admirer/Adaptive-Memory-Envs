import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser

def plot_and_save_curves(all_histories, base_run_id, num_runs):
    """
    Averages the histories of all runs and plots the results.
    
    Arguments:
        all_histories {list} -- A list containing the history dictionary of each run.
        base_run_id {str} -- The base name for saving the plot files.
        num_runs {int} -- The number of runs to average over.
    """
    print("Averaging and plotting learning curves...")
    
    # Create a directory for plots if it doesn't exist
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # Metrics to plot
    metrics_to_plot = list(all_histories[0].keys())

    for metric in metrics_to_plot:
        # Collect all runs for the current metric
        metric_runs = [h[metric] for h in all_histories if metric in h and h[metric]]
        
        if not metric_runs:
            print(f"Skipping metric '{metric}' as no data was recorded.")
            continue
            
        # Pad runs to the same length for averaging, because some runs might be shorter
        max_len = max(len(run) for run in metric_runs)
        padded_runs = []
        for run in metric_runs:
            # Use np.nan for padding to ignore these values in nanmean/nanstd
            padded_run = np.pad(run, (0, max_len - len(run)), 'constant', constant_values=np.nan)
            padded_runs.append(padded_run)
            
        # Calculate mean and std deviation across runs, ignoring NaNs
        mean_curve = np.nanmean(padded_runs, axis=0)
        std_curve = np.nanstd(padded_runs, axis=0)
        
        # Plotting
        plt.figure(figsize=(12, 8))
        x_axis = np.arange(max_len)
        
        # Plot mean curve
        plt.plot(x_axis, mean_curve, label='Mean')
        
        # Plot confidence interval (mean +/- std)
        plt.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3, label='Std Dev')
        
        plt.title(f'Average {metric.replace("_", " ").title()} over {num_runs} runs')
        plt.xlabel('Update Step')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_filename = f'./plots/{base_run_id}_{metric}_len5_num5.png'
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot to {plot_filename}")

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=./configs/endless_tmaze.yaml            Path to the yaml config file [default: ./configs/endless_tmaze.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    base_run_id = options["--run-id"]
    cpu = options["--cpu"]
    
    # Parse the yaml config file.
    config = YamlParser(options["--config"]).get_config()

    # Determine the device to be used for training
    if not cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # --- Training Loop for Multiple Runs ---
    num_runs = 3
    all_runs_histories = []

    for i in range(num_runs):
        run_id = f"{base_run_id}_run_{i+1}"
        print(f"\n--- Starting Run {i+1}/{num_runs} with ID: {run_id} ---\n")
        
        # Initialize the PPO trainer and commence training
        trainer = PPOTrainer(config, run_id=run_id, device=device)
        history = trainer.run_training()
        all_runs_histories.append(history)
        trainer.close()

    # --- Averaging and Plotting ---
    if all_runs_histories:
        plot_and_save_curves(all_runs_histories, base_run_id, num_runs)

if __name__ == "__main__":
    main()