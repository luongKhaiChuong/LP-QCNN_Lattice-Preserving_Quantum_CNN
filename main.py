import time
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import torch
import numpy as np
import argparse
import json

import config
from data_utils import get_dataloaders
from model import SimpleQuantumCNN
from train import train_engine

def get_experiments_list():
    return [
        {
            "id": "Experiment_01_MNIST_Epoch",
            "dataset": "MNIST",
            "classes": None,
            "mode": "epoch",
            "max_duration": config.MAX_EPOCHS,
            "batch_size": config.BATCH_SIZE_EPOCH,
            "n_classes": 10,
            "grayscale": True,
            "in_channels": 1
        },
        {
            "id": "Experiment_02_MNIST_Iter",
            "dataset": "MNIST",
            "classes": None,
            "mode": "iter",
            "max_duration": config.MAX_ITERS,
            "batch_size": config.BATCH_SIZE_ITER,
            "n_classes": 10,
            "grayscale": True,
            "in_channels": 1
        },
        {
            "id": "Experiment_03_FashionMNIST",
            "dataset": "FashionMNIST",
            "classes": None,
            "mode": "epoch",
            "max_duration": config.MAX_EPOCHS,
            "batch_size": config.BATCH_SIZE_EPOCH,
            "n_classes": 10,
            "grayscale": True,
            "in_channels": 1
        },
        {
            "id": "Experiment_04_CIFAR10",
            "dataset": "CIFAR10",
            "classes": None,
            "mode": "epoch",
            "max_duration": config.MAX_EPOCHS,
            "batch_size": config.BATCH_SIZE_EPOCH,
            "n_classes": 10,
            "grayscale": False,
            "in_channels": 3
        }
    ]

# --- PLOTTING HELPERS ---
def plot_mean_and_shade(ax, x, y_data, color, label, marker, axis=0):
    y_np = np.array(y_data)
    mean = np.mean(y_np, axis=axis)
    std = np.std(y_np, axis=axis)
    ax.plot(x, mean, color=color, label=label, marker=marker, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

def save_history_to_json(history, save_path):
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)

def plot_sensitivity_analysis(results, metric_key, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    x_values = results['num_filters']
    
    def ensure_2d(data):
        if not data: return np.array([])
        return np.array(data)

    y_trainable = ensure_2d(results[f'trainable_{metric_key}'])
    y_frozen = ensure_2d(results[f'frozen_{metric_key}'])
    
    if y_trainable.ndim == 1:
        mean_t, std_t = y_trainable, np.zeros_like(y_trainable)
        mean_f, std_f = y_frozen, np.zeros_like(y_frozen)
    else:
        mean_t = np.mean(y_trainable, axis=1)
        std_t = np.std(y_trainable, axis=1)
        mean_f = np.mean(y_frozen, axis=1)
        std_f = np.std(y_frozen, axis=1)
    
    ax.plot(x_values, mean_t, 'o-', color='blue', label='Trainable AQC')
    ax.fill_between(x_values, mean_t - std_t, mean_t + std_t, color='blue', alpha=0.15)
    
    ax.plot(x_values, mean_f, 'x--', color='orange', label='Frozen Baseline')
    ax.fill_between(x_values, mean_f - std_f, mean_f + std_f, color='orange', alpha=0.15)

    plt.xlabel('Number of Filters')
    plt.ylabel(ylabel)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_metric_curve(steps, data_trainable, data_frozen, metric_name, num_filters, mode, n_trials, save_path):
    plt.figure()
    ax = plt.gca()
    title_suffix = f"(Mean of {n_trials} Trials)" if n_trials > 1 else "(Single Run)"
    
    if n_trials > 1:
        plot_mean_and_shade(ax, steps, data_trainable, 'blue', 'Trainable (Mean)', 'o', axis=0)
        plot_mean_and_shade(ax, steps, data_frozen, 'orange', 'Frozen (Mean)', 'x', axis=0)
    else:
        ax.plot(steps, data_trainable[0], label='Trainable', marker='o', color='blue')
        ax.plot(steps, data_frozen[0], label='Frozen', marker='s', linestyle='--', color='orange')

    
    if mode == 'iter':
            ax.xaxis.set_major_locator(MultipleLocator(20)) 
            plt.xlabel('Steps (Iterations)')
    else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Epochs')
            
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def run_experiment_suite(exp_config):
    exp_id = exp_config["id"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_root_dir = os.path.join(config.OUTPUT_DIR, f"{exp_id}_{timestamp}")
    os.makedirs(exp_root_dir, exist_ok=True)
    
    n_trials = config.NUM_TRIALS if exp_config["mode"] == 'iter' else 1
    
    print(f"\n{'='*70}\n STARTING: {exp_id}\n Mode: {exp_config['mode'].upper()} | Trials: {n_trials}\n Dir: {exp_root_dir}\n{'='*70}")
    
    train_loader, val_loader = get_dataloaders(
        dataset_name=exp_config["dataset"],
        batch_size=exp_config["batch_size"],
        classes=exp_config["classes"],
        grayscale=exp_config.get("grayscale", True)
    )
    
    sensitivity_results = {
        'num_filters': [],
        'trainable_acc': [], 'trainable_loss': [],
        'frozen_acc': [], 'frozen_loss': []
    }
    
    iter_summary_data = []
    epoch_summary_data = []
    
    is_grayscale = exp_config.get("grayscale", True)
    current_search_range = config.FILTER_RANGE_GRAYSCALE if is_grayscale else config.FILTER_RANGE_RGB
    print(f">>> Search Range: {current_search_range}")

    for num_filters in current_search_range:
        in_channels = exp_config.get("in_channels", 1)
        if in_channels > 1 and num_filters % in_channels != 0:
            continue
        
        print(f"\n>>> Filters: {num_filters}")
        sub_dir = os.path.join(exp_root_dir, f"Filters_{num_filters}")
        os.makedirs(sub_dir, exist_ok=True)
        
        filter_logs = {
            'trainable_acc_runs': [], 'frozen_acc_runs': [],
            'trainable_loss_runs': [], 'frozen_loss_runs': [] 
        }
        
        best_acc_t_list, best_loss_t_list = [], []
        best_acc_f_list, best_loss_f_list = [], []
        common_steps = []

        # === LOOP TRIALS ===
        for trial in range(n_trials):
            print(f"   [Trial {trial + 1}/{n_trials}]")
            
            # Trainable
            model_t = SimpleQuantumCNN(n_filters=num_filters, n_classes=exp_config["n_classes"], in_channels=in_channels)
            hist_t = train_engine(model_t, train_loader, val_loader, 
                                  os.path.join(sub_dir, f"model_trainable_trial{trial}.pth"), 
                                  mode=exp_config["mode"], max_duration=exp_config["max_duration"])
            save_history_to_json(hist_t, os.path.join(sub_dir, f"metrics_trainable_trial{trial}.json"))
            
            # Frozen
            model_f = SimpleQuantumCNN(n_filters=num_filters, n_classes=exp_config["n_classes"], in_channels=in_channels)
            model_f.hamiltonian_params.requires_grad = False
            if isinstance(model_f.alpha, torch.Tensor): model_f.alpha.requires_grad = False
            hist_f = train_engine(model_f, train_loader, val_loader, 
                                  os.path.join(sub_dir, f"model_frozen_trial{trial}.pth"), 
                                  mode=exp_config["mode"], max_duration=exp_config["max_duration"])
            save_history_to_json(hist_f, os.path.join(sub_dir, f"metrics_frozen_trial{trial}.json"))
            
            filter_logs['trainable_acc_runs'].append(hist_t['val_acc'])
            filter_logs['frozen_acc_runs'].append(hist_f['val_acc'])
            filter_logs['trainable_loss_runs'].append(hist_t['val_loss'])
            filter_logs['frozen_loss_runs'].append(hist_f['val_loss'])
            
            best_acc_t_list.append(max(hist_t['val_acc']))
            best_loss_t_list.append(min(hist_t['val_loss']) if hist_t['val_loss'] else 0)
            best_acc_f_list.append(max(hist_f['val_acc']))
            best_loss_f_list.append(min(hist_f['val_loss']) if hist_f['val_loss'] else 0)
            
            if len(hist_t['step']) >= len(common_steps):
                common_steps = hist_t['step']
            
            if exp_config["mode"] == 'epoch':
                idx_t = np.argmax(hist_t['val_acc'])
                idx_f = np.argmax(hist_f['val_acc'])
                
                epoch_summary_data.append({
                    "filters": num_filters,
                    "trainable": {
                        "best_epoch": hist_t['step'][idx_t],
                        "val_acc": hist_t['val_acc'][idx_t],
                        "val_loss": hist_t['val_loss'][idx_t]
                    },
                    "frozen": {
                        "best_epoch": hist_f['step'][idx_f],
                        "val_acc": hist_f['val_acc'][idx_f],
                        "val_loss": hist_f['val_loss'][idx_f]
                    }
                })

        if exp_config["mode"] == 'iter':
            def fmt_stat(data_list, is_acc=True):
                mean = np.mean(data_list)
                std = np.std(data_list)
                if is_acc: return f"{mean*100:.2f} ± {std*100:.2f}%"
                else: return f"{mean:.4f} ± {std:.4f}"

            iter_summary_data.append({
                "filters": num_filters,
                "trainable": {
                    "val_acc": fmt_stat(best_acc_t_list, True),
                    "val_loss": fmt_stat(best_loss_t_list, False)
                },
                "frozen": {
                    "val_acc": fmt_stat(best_acc_f_list, True),
                    "val_loss": fmt_stat(best_loss_f_list, False)
                }
            })

        plot_metric_curve(common_steps, filter_logs['trainable_acc_runs'], filter_logs['frozen_acc_runs'], 
                          'Val Accuracy', num_filters, exp_config["mode"], n_trials, os.path.join(sub_dir, "learning_curve_accuracy.png"))
        plot_metric_curve(common_steps, filter_logs['trainable_loss_runs'], filter_logs['frozen_loss_runs'], 
                          'Val Loss', num_filters, exp_config["mode"], n_trials, os.path.join(sub_dir, "learning_curve_loss.png"))

        sensitivity_results['num_filters'].append(num_filters)
        sensitivity_results['trainable_acc'].append(best_acc_t_list)
        sensitivity_results['trainable_loss'].append(best_loss_t_list)
        sensitivity_results['frozen_acc'].append(best_acc_f_list)
        sensitivity_results['frozen_loss'].append(best_loss_f_list)

    # Final Summary & JSON Save
    if len(sensitivity_results['num_filters']) > 0:
        plot_sensitivity_analysis(sensitivity_results, 'acc', 'Best val Acc', os.path.join(exp_root_dir, "analysis_val_accuracy.png"))
        plot_sensitivity_analysis(sensitivity_results, 'loss', 'Best val Loss', os.path.join(exp_root_dir, "analysis_val_loss.png"))
        
        with open(os.path.join(exp_root_dir, "summary_raw_data.json"), 'w') as f:
            json.dump(sensitivity_results, f, indent=4)
            
        if exp_config["mode"] == 'iter':
            with open(os.path.join(exp_root_dir, "summary_iter_stats.json"), 'w') as f:
                json.dump(iter_summary_data, f, indent=4)
            print(f"   -> Saved Summary Iter Stats")
            
        if exp_config["mode"] == 'epoch':
            with open(os.path.join(exp_root_dir, "summary_epoch_best.json"), 'w') as f:
                json.dump(epoch_summary_data, f, indent=4)
            print(f"   -> Saved Summary Best Epochs")
            
        print(f"Experiment {exp_id} COMPLETED.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()
    config.set_debug_mode(args.debug)
    for exp in get_experiments_list():
        run_experiment_suite(exp)