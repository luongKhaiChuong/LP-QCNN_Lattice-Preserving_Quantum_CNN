import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import config
import os
import numpy as np
from tqdm import tqdm
from trainer import train_experiment

matplotlib.use('Agg') 

def run_phase_1(best_configs):
    """
    Executes Phase 1: Deep Analysis & Physical Metrics.

    - Experimental Goal:
    Perform a comprehensive training run (20 epochs) using the optimal configurations 
    found from various trials. This phase analyzes the model's convergence and physical properties 
    (BCH Error, Fisher Rank) over time.

    - Process:
        + Loads the best configuration from Phase 2.
        + Trains for extended epochs (20) on the full dataset.
        + Computes BCH Error and Fisher Rank at each epoch.
        + Saves the trained model weights.
        + Plots: Loss/Accuracy Evolution, BCH Error Evolution, Fisher Rank Evolution.

    Args:
        best_configs (dict): containing optimal parameters.
    """
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 1 ({config.MODE}): DEEP ANALYSIS (20 EPOCHS)")
    config.LOGGER.info("="*40)
    
    phase1_results = []
    
    if config.MODE == 'DUMMY':
        data_size = 100
        epochs = 3
    else:
        data_size = 'fullset'
        epochs = 20
    for ds_name, best_cfg in best_configs.items():
        
        print (best_configs.items())
        config.LOGGER.info(f">>> Deep Training {ds_name} with Config: {best_cfg}")
        
        run_cfg = best_cfg.copy()
        run_cfg['data_size'] = data_size
        run_cfg['epochs'] = epochs
        
        save_path = f"model_P1_{ds_name}_{config.MODE}.pth"
        
        # Enable analyze_physics=True for detailed metrics
        hist, final_acc, _ = train_experiment(run_cfg, f"P1_{ds_name}_FINAL", 
                                              analyze_physics=True, save_model_path=save_path)
        phase1_results.append(hist)
        
        epochs_x = range(1, len(hist['train_loss']) + 1)
        
        # --- Plot 1: Performance Evolution ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(epochs_x, hist['train_loss'], label='Train Loss', linestyle='--', color='tab:blue', marker='.')
        ax1.plot(epochs_x, hist['val_loss'], label='Val Loss', linestyle='-', color='tab:orange', marker='.')
        ax1.set_title(f"Loss Evolution ({ds_name})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Accuracy
        ax2.plot(epochs_x, hist['train_acc'], label='Train Acc', linestyle='--', color='tab:green', marker='.')
        ax2.plot(epochs_x, hist['val_acc'], label='Val Acc', linestyle='-', color='tab:red', marker='.')
        ax2.set_title(f"Accuracy Evolution ({ds_name})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        filename_perf = f"P1_{ds_name}_Perf_{config.MODE}.png"
        plt.savefig(os.path.join(config.RUN_DIR, filename_perf))
        plt.close()
        
        # --- Plot 2: BCH Error ---
        plt.figure()
        plt.plot(epochs_x, hist['bch'], label='BCH Error', color='crimson', marker='o')
        plt.title(f"BCH Error Evolution ({ds_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Error Value")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(config.RUN_DIR, f"P1_{ds_name}_BCH_{config.MODE}.png"))
        plt.close()
        
        # --- Plot 3: Fisher Rank ---
        plt.figure()
        plt.plot(epochs_x, hist['fisher_rank'], label='Fisher Rank', color='forestgreen', marker='s')
        plt.title(f"Fisher Rank Evolution ({ds_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Effective Rank")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(config.RUN_DIR, f"P1_{ds_name}_Fisher_{config.MODE}.png"))
        plt.close()

    config.GLOBAL_RESULTS['phase_1'] = phase1_results

def run_phase_2(best_configs):
    """
    Executes Phase 2: Data Efficiency Analysis.
    """
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 2 ({config.MODE}): DATA EFFICIENCY")
    config.LOGGER.info("="*40)
    
    phase2_results = []
    sizes = [10, 20, 50] if config.MODE == 'DUMMY' else [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 'fullset']
    epochs = 5
    times = 5

    original_seed = getattr(config, 'SEED', 42)

    for ds_name, val in best_configs.items():
        if isinstance(val, dict):
            base_cfg = val.copy()
            base_cfg['epochs'] = epochs
            config.LOGGER.info(f">>> Using Optimized Config for {ds_name}: {base_cfg}")
        else:
            base_cfg = {'dataset': ds_name, 'epochs': epochs, 'time': 1.0, 'time_steps': 1, 'n_filters': val}
            config.LOGGER.info(f">>> Using Basic Config for {ds_name}: {base_cfg}")

        size_metrics = {
            'dataset': ds_name, 
            'sizes': [], 
            'accuracies_mean': [], 
            'accuracies_std': [], 
            'losses_mean': []
        }
        
        pbar = tqdm(sizes, desc=f"P2 {ds_name} Data Sizes")
        for sz in pbar:
            run_cfg = base_cfg.copy()
            run_cfg['data_size'] = sz
            
            sz_accuracies = []
            sz_losses = []
            
            for t in range(times):
                config.SEED = original_seed + t
                hist, acc, _ = train_experiment(run_cfg, f"P2_{ds_name}_Size_{sz}_Run_{t+1}")
                sz_accuracies.append(acc)
                sz_losses.append(hist['val_loss'][-1])
            
            mean_acc = np.mean(sz_accuracies)
            std_acc = np.std(sz_accuracies)
            mean_loss = np.mean(sz_losses)
            
            size_metrics['sizes'].append(str(sz))
            size_metrics['accuracies_mean'].append(mean_acc)
            size_metrics['accuracies_std'].append(std_acc)
            size_metrics['losses_mean'].append(mean_loss)
            
        phase2_results.append(size_metrics)
        
        config.SEED = original_seed
        
        plt.figure(figsize=(8, 5))
        
        x_indices = np.arange(len(size_metrics['sizes']))
        y_mean = np.array(size_metrics['accuracies_mean'])
        y_std = np.array(size_metrics['accuracies_std'])
        
        plt.plot(x_indices, y_mean, marker='o', color='purple', label='Mean Accuracy')
        plt.fill_between(x_indices, y_mean - y_std, y_mean + y_std, color='purple', alpha=0.2, label='Standard Deviation')
        
        plot_labels = ['60000' if str(s) == 'fullset' else str(s) for s in size_metrics['sizes']]
        plt.xticks(x_indices, plot_labels)
        
        plt.title(f"Data Necessity ({ds_name})")
        plt.xlabel("Data Size")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.grid(True)
        
        filename = f"P2_{ds_name}_Eff_{config.MODE}.png"
        plt.savefig(os.path.join(config.RUN_DIR, filename))
        plt.close()

    config.GLOBAL_RESULTS['phase_2'] = phase2_results

def run_phase_3():
    """
    Executes Phase 3: Binary Stability Test.

    - Experimental Goal:
    Verify the model's stability and consistency on a simplified binary classification task (0 vs 1).
    Running multiple times helps assess the variance due to initialization.

    - Process:
        + Filters datasets to only include class 0 and class 1.
        + Runs the model multiple times (e.g., 10 times) with a standard configuration.
        + Collects performance metrics to analyze stability.
    """
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 3 ({config.MODE}): BINARY STABILITY")
    config.LOGGER.info("="*40)
    
    phase3_data = {}
    for ds_name in ['MNIST', 'FashionMNIST']:
        config.LOGGER.info(f">>> PROCESSING: {ds_name} (Binary)")
        cfg = {'dataset': ds_name, 'data_size': 'fullset', 'epochs': 1, 'batch_size': 25, 
               'n_filters': 1, 'time': 1.0, 'time_steps': 1, 'binary_mode': True}
        
        num_runs = 5 if config.MODE == 'DUMMY' else 10
        runs_data = []
        accuracies = []
        
        pbar = tqdm(range(num_runs), desc=f"P3 {ds_name} Runs")
        for i in pbar:
            hist, acc, _ = train_experiment(cfg, f"P3_{ds_name}_Run_{i+1}", max_steps=200) #batch_size 25, iteration 200
            runs_data.append({
                'run_id': i+1, 
                'final_acc': acc, 
                'final_loss': hist['val_loss'][-1], 
                'loss_curve': hist['train_loss']
            })
            accuracies.append(acc)
            
        phase3_data[ds_name] = runs_data
        
        # Calculate mean and standard deviation
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        # Print the aggregated result
        config.LOGGER.info(f"Result for {ds_name} -> Accuracy: {mean_acc:.2f} +- {std_acc:.2f}%")
    
    config.GLOBAL_RESULTS['phase_3_binary_stability'] = phase3_data

# def run_phase_4(best_configs):
#     """
#     Executes Phase 4: Noise Resilience Analysis.
    
#     - Experimental Goal:
#     Evaluate the model's accuracy degradation under various levels of Depolarizing noise.
#     Uses 'default.mixed' device in PennyLane for density matrix simulations.
#     """
#     config.LOGGER.info("\n" + "="*40)
#     config.LOGGER.info(f"PHASE 4 ({config.MODE}): NOISE RESILIENCE")
#     config.LOGGER.info("="*40)
    
#     phase4_results = []
    
#     noise_levels = [0.0, 0.01, 0.05] if config.MODE == 'DUMMY' else [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
#     epochs = 2 if config.MODE == 'DUMMY' else 5
    
#     for ds_name, val in best_configs.items():
#         if isinstance(val, dict):
#             base_cfg = val.copy()
#             base_cfg['epochs'] = epochs
#         else:
#             base_cfg = {'dataset': ds_name, 'epochs': epochs, 'time': 1.0, 'time_steps': 1, 'n_filters': val}
            
#         base_cfg['data_size'] = 5 if config.MODE == 'DUMMY' else 'fullset' 
#         config.LOGGER.info(f">>> Processing Noise Resilience for {ds_name}")
        
#         noise_metrics = {
#             'dataset': ds_name,
#             'noise_levels': noise_levels,
#             'accuracies': []
#         }
        
#         pbar = tqdm(noise_levels, desc=f"P4 {ds_name} Noise Levels")
#         for noise in pbar:
#             run_cfg = base_cfg.copy()
#             run_cfg['noise_prob'] = noise
            
#             hist, final_acc, _ = train_experiment(run_cfg, f"P4_{ds_name}_Noise_{noise}")
#             noise_metrics['accuracies'].append(final_acc)
            
#         phase4_results.append(noise_metrics)
        
#         plt.figure(figsize=(8, 5))
#         plt.plot(noise_levels, noise_metrics['accuracies'], marker='o', color='darkorange', linewidth=2)
#         plt.title(f"Noise Resilience - Accuracy Degradation ({ds_name})")
#         plt.xlabel("Depolarizing Noise Probability ($p$)")
#         plt.ylabel("Validation Accuracy")
#         plt.grid(True, linestyle='--', alpha=0.7)
        
#         plt.xticks(noise_levels, [f"{int(n*100)}%" for n in noise_levels])
        
#         filename = f"P4_{ds_name}_Noise_Resilience_{config.MODE}.png"
#         plt.savefig(os.path.join(config.RUN_DIR, filename))
#         plt.close()

#     config.GLOBAL_RESULTS['phase_4_noise'] = phase4_results