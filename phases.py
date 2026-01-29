import itertools
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import config
import os
import numpy as np
from tqdm import tqdm
from trainer import train_experiment

def run_phase_1():
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 1 ({config.MODE}): OPTIMAL FILTER SEARCH")
    config.LOGGER.info("="*40)
    
    phase1_results = []
    
    if config.MODE == 'DUMMY':
        datasets_config = {'MNIST': range(1, 3), 'FashionMNIST': range(1, 3)}
        data_size, epochs = 1000, 2
    else:
        datasets_config = {'MNIST': range(1, 4), 'FashionMNIST': range(1, 7)}
        data_size, epochs = 25000, 5
    
    best_configs = {}
    
    for ds_name, filter_range in datasets_config.items():
        best_acc = -1; best_loss = float('inf'); best_filter = 1
        filters_list, acc_list, time_list = [], [], []

        pbar = tqdm(filter_range, desc=f"P1 {ds_name} Filters")
        
        for f in pbar:
            cfg = {'dataset': ds_name, 'data_size': data_size, 'epochs': epochs, 'time': 3.0, 'time_steps': 3, 'n_filters': f}
            hist, final_acc, t_time = train_experiment(cfg, f"P1_{ds_name}_F{f}")
            phase1_results.append(hist)
            filters_list.append(f); acc_list.append(final_acc); time_list.append(t_time)
            
            final_loss = hist['val_loss'][-1] if hist['val_loss'] else float('inf')
            if final_acc > best_acc:
                best_acc = final_acc; best_loss = final_loss; best_filter = f
            elif final_acc == best_acc:
                if final_loss < best_loss: best_loss = final_loss; best_filter = f
        
        best_configs[ds_name] = best_filter
        config.LOGGER.info(f"--> Best {ds_name} Filter: {best_filter} (Acc: {best_acc:.4f})")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(filters_list, acc_list, marker='o', color='blue')
        ax1.set_title(f"Acc vs Filters ({ds_name})"); ax1.set_xlabel("Filters"); ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2.bar(filters_list, time_list, color='orange', alpha=0.7)
        ax2.set_title(f"Time vs Filters ({ds_name})"); ax2.set_xlabel("Filters"); ax2.grid(True, axis='y')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        filename = f"P1_{ds_name}_Tradeoff_{config.MODE}.png"
        plt.savefig(os.path.join(config.RUN_DIR, filename))
        plt.close()
        
    config.GLOBAL_RESULTS['phase_1'] = phase1_results
    return best_configs

def run_phase_2(p1_best_configs):
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 2 ({config.MODE}): DYNAMICS GRID SEARCH")
    config.LOGGER.info("="*40)
    
    phase2_results = []
    best_phase2 = {}
    
    if config.MODE == 'DUMMY':
        times = [1.0]
        grids = {'MNIST': {'time_steps': [1, 2]}, 'FashionMNIST': {'time_steps': [1, 2]}}
        data_size, epochs = 1000, 3
    else:
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        grids = {'MNIST': {'time_steps': [1, 2, 4]}, 'FashionMNIST': {'time_steps': [1, 2, 4, 8]}}
        data_size, epochs = 25000, 5
    
    for ds_name, params in grids.items():
        best_acc = -1; best_cfg = None; records = []
        optimal_filter = p1_best_configs[ds_name]
        config.LOGGER.info(f">>> Dataset {ds_name}: Locking n_filters={optimal_filter}")
        
        combinations = list(itertools.product(times, params['time_steps'], [optimal_filter]))
        
        pbar = tqdm(combinations, desc=f"P2 {ds_name} Grid")
        
        for t, s, f in pbar:
            cfg = {'dataset': ds_name, 'data_size': data_size, 'epochs': epochs, 'time': t, 'time_steps': s, 'n_filters': f}
            hist, final_acc, _ = train_experiment(cfg, f"P2_{ds_name}_T{t}_S{s}_F{f}")
            phase2_results.append(hist); records.append({'time': t, 'steps': s, 'filters': f, 'acc': final_acc})
            if final_acc > best_acc: best_acc = final_acc; best_cfg = cfg
        
        if best_cfg is None: best_cfg = cfg
        best_phase2[ds_name] = best_cfg
        config.LOGGER.info(f"--> Best {ds_name} P2 Config: {best_cfg}")

        if not records: continue
        unique_times = sorted(list(set(r['time'] for r in records)))
        
        color_map = plt.get_cmap('tab10') 
        linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
        markers = ['o', 's', '^', 'D', 'v', '<', '>', '*', 'P', 'X']
        style_cycle = itertools.cycle(itertools.product(linestyles, markers))
        
        plt.figure(figsize=(10, 7))
        for t_idx, t in enumerate(unique_times):
            subset = [r for r in records if r['time'] == t]
            if not subset: continue
            subset.sort(key=lambda x: x['steps'])
            y_vals = [r['acc'] for r in subset]
            x_vals_str = [str(r['steps']) for r in subset]
            c = color_map(t_idx % 10)
            ls, mk = next(style_cycle)
            plt.plot(x_vals_str, y_vals, marker=mk, linestyle=ls, color=c, 
                     label=f"Time={t}", linewidth=2, markersize=8, alpha=0.9)
                      
        plt.title(f"Dynamics Analysis ({ds_name}) [Filter={optimal_filter}]")
        plt.xlabel("Time Steps (Equally Spaced Categories)")
        plt.ylabel("Validation Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), title="Duration (Time)")
        plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
        filename = f"P2_{ds_name}_Scatter_{config.MODE}.png"
        plt.savefig(os.path.join(config.RUN_DIR, filename))
        plt.close()

    config.GLOBAL_RESULTS['phase_2'] = phase2_results
    return best_phase2

def run_phase_3(p2_best_configs):
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 3 ({config.MODE}): DEEP ANALYSIS (20 EPOCHS)")
    config.LOGGER.info("="*40)
    
    phase3_results = []
    
    if config.MODE == 'DUMMY':
        data_size = 1000
        epochs = 5
    else:
        data_size = 'fullset'
        epochs = 20
        
    for ds_name, best_cfg in p2_best_configs.items():
        config.LOGGER.info(f">>> Deep Training {ds_name} with Config: {best_cfg}")
        
        run_cfg = best_cfg.copy()
        run_cfg['data_size'] = data_size
        run_cfg['epochs'] = epochs
        
        save_path = f"model_P3_{ds_name}_{config.MODE}.pth"
        
        hist, final_acc, _ = train_experiment(run_cfg, f"P3_{ds_name}_FINAL", analyze_physics=True, save_model_path=save_path)
        phase3_results.append(hist)
        
        epochs_x = range(1, len(hist['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs_x, hist['train_loss'], label='Train Loss', linestyle='--', color='tab:blue', marker='.')
        ax1.plot(epochs_x, hist['val_loss'], label='Val Loss', linestyle='-', color='tab:orange', marker='.')
        ax1.set_title(f"Loss Evolution ({ds_name})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2.plot(epochs_x, hist['train_acc'], label='Train Acc', linestyle='--', color='tab:green', marker='.')
        ax2.plot(epochs_x, hist['val_acc'], label='Val Acc', linestyle='-', color='tab:red', marker='.')
        ax2.set_title(f"Accuracy Evolution ({ds_name})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        filename_perf = f"P3_{ds_name}_Perf_{config.MODE}.png"
        plt.savefig(os.path.join(config.RUN_DIR, filename_perf))
        plt.close()
        
        # BCH Error
        plt.figure()
        plt.plot(epochs_x, hist['bch'], label='BCH Error', color='crimson', marker='o')
        plt.title(f"BCH Error Evolution ({ds_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Error Value")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(config.RUN_DIR, f"P3_{ds_name}_BCH_{config.MODE}.png"))
        plt.close()
        
        # Fisher Rank
        plt.figure()
        plt.plot(epochs_x, hist['fisher_rank'], label='Fisher Rank', color='forestgreen', marker='s')
        plt.title(f"Fisher Rank Evolution ({ds_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Effective Rank")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(config.RUN_DIR, f"P3_{ds_name}_Fisher_{config.MODE}.png"))
        plt.close()

    config.GLOBAL_RESULTS['phase_3'] = phase3_results

def run_phase_4(best_configs):
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 4 ({config.MODE}): DATA EFFICIENCY")
    config.LOGGER.info("="*40)
    
    phase4_results = []
    sizes = [1000, 2000, 5000] if config.MODE == 'DUMMY' else [1000, 2000, 5000, 10000, 20000, 50000, 'fullset']
    epochs = 5
    
    for ds_name, val in best_configs.items():
        if isinstance(val, dict):
            base_cfg = val.copy()
            base_cfg['epochs'] = epochs
            config.LOGGER.info(f">>> Using Optimized Config from P2 for {ds_name}: {base_cfg}")
        else:
            base_cfg = {'dataset': ds_name, 'epochs': epochs, 'time': 1.0, 'time_steps': 1, 'n_filters': val}
            config.LOGGER.info(f">>> Using Basic Config (Only Filter from P1) for {ds_name}: {base_cfg}")


        size_metrics = {'dataset': ds_name, 'sizes': [], 'accuracies': [], 'losses': []}
        
        pbar = tqdm(sizes, desc=f"P4 {ds_name} Data Sizes")
        for sz in pbar:
            run_cfg = base_cfg.copy()
            run_cfg['data_size'] = sz
            
            hist, acc, _ = train_experiment(run_cfg, f"P4_{ds_name}_Size_{sz}")
            
            size_metrics['sizes'].append(str(sz))
            size_metrics['accuracies'].append(acc)
            size_metrics['losses'].append(hist['val_loss'][-1])
            
        phase4_results.append(size_metrics)
        
        plt.figure(figsize=(8, 5))
        plt.plot(size_metrics['sizes'], size_metrics['accuracies'], marker='o', color='purple')
        plt.title(f"Data Necessity ({ds_name})")
        plt.xlabel("Data Size")
        plt.ylabel("Validation Accuracy")
        plt.grid(True)
        
        filename = f"P4_{ds_name}_Eff_{config.MODE}.png"
        plt.savefig(os.path.join(config.RUN_DIR, filename))
        plt.close()

    config.GLOBAL_RESULTS['phase_4'] = phase4_results

def run_phase_5():
    config.LOGGER.info("\n" + "="*40)
    config.LOGGER.info(f"PHASE 5 ({config.MODE}): BINARY STABILITY")
    config.LOGGER.info("="*40)
    
    phase5_data = {}
    for ds_name in ['MNIST', 'FashionMNIST']:
        config.LOGGER.info(f">>> PROCESSING: {ds_name} (Binary)")
        cfg = {'dataset': ds_name, 'data_size': 'fullset', 'epochs': 1, 'batch_size': 25, 
               'n_filters': 1, 'time': 1.0, 'time_steps': 1, 'binary_mode': True}
        num_runs = 2 if config.MODE == 'DUMMY' else 10
        runs_data = []
        
        pbar = tqdm(range(num_runs), desc=f"P5 {ds_name} Runs")
        for i in pbar:
            hist, acc, _ = train_experiment(cfg, f"P5_{ds_name}_Run_{i+1}", max_steps=200)
            runs_data.append({'run_id': i+1, 'final_acc': acc, 'final_loss': hist['val_loss'][-1], 'loss_curve': hist['train_loss']})
        phase5_data[ds_name] = runs_data
    config.GLOBAL_RESULTS['phase_5'] = phase5_data