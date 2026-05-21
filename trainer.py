import torch
import torch.nn as nn
import time
import config
import os
from tqdm import tqdm
from utils import seed_everything  # Renamed from set_seed to match utils.py
from data import get_dataloaders
from model import LPQCNN
from analyzer import Analyzer

def train_experiment(config_dict, run_name, analyze_physics=False, max_steps=None, save_model_path=None):
    """
    Executes the complete experimental training protocol for the Lattice-Preserving QCNN.

    This function handles:
    1.  **Initialization:** Setting random seeds, initializing the model, optimizer, and loss function.
    2.  **Data Loading:** Retrieving Train/Test DataLoaders based on configuration.
    3.  **Training Loop:** Iterating through epochs and batches to optimize model parameters.
    4.  **Validation:** Evaluating model performance on the test set.
    5.  **Analysis:** Optionally computing physics-based metrics (BCH Error, Fisher Rank).
    6.  **Logging & Persistence:** Tracking history and saving the final model state.

    Args:
        config_dict (dict): Dictionary containing hyperparameters (e.g., 'lr', 'epochs', 'n_filters', 'time').
        run_name (str): Unique identifier for the current experiment run (used for logging).
        analyze_physics (bool): If True, computes BCH Error and Fisher Rank at the end of each epoch.
                                Note: This significantly increases computation time.
        max_steps (int, optional): limit the number of training batches (useful for debugging).
        save_model_path (str, optional): Relative path to save the trained model state dict.

    Returns:
        tuple: (history_dict, final_validation_accuracy, total_training_time)
    """
    # 1. Reproducibility Setup
    seed_everything(config.SEED)
    
    # 2. Data Preparation
    use_binary_mode = config_dict.get('binary_mode', False)
    batch_size = config_dict.get('batch_size', 128)
    
    train_loader, test_loader = get_dataloaders(
        dataset_name=config_dict['dataset'], 
        data_size=config_dict['data_size'], 
        batch_size=batch_size,
        binary_mode=use_binary_mode
    )
    
    # 3. Model & Optimizer Initialization
    num_classes = 2 if use_binary_mode else 10
    noise_prob = config_dict.get('noise_prob', 0.0)
    
    model = LPQCNN(
        n_filters=config_dict['n_filters'], 
        time_dyn=config_dict['time'], 
        time_steps=config_dict['time_steps'], 
        n_classes=num_classes,
        noise_prob=noise_prob
    ).to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize History Tracker
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'bch': [], 'fisher_rank': [], 
        'time': 0, 'config': config_dict
    }
    
    if config.LOGGER:
        config.LOGGER.info(f">>> START RUN: {run_name} | Configuration: {config_dict}")
    
    # 4. Training Loop
    global_step = 0
    stop_training = False
    start_time = time.time()
    
    # TQDM Progress Bar for Epochs
    epoch_pbar = tqdm(range(config_dict['epochs']), desc=f"Training {run_name}", position=0, leave=True)
    
    for epoch_idx in epoch_pbar:
        if stop_training: break
        
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0
        total_train_samples = 0
        
        # TQDM Progress Bar for Batches (Inner loop)
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}", leave=False, position=1)
        
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward Pass & Optimization
            loss.backward()
            optimizer.step()
            
            # Metrics Accumulation
            batch_size_current = inputs.size(0)
            running_train_loss += loss.item() * batch_size_current
            running_train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_train_samples += batch_size_current
            
            # Update Batch Progress Bar
            batch_pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
            # Debugging/Safety Stop
            global_step += 1
            if max_steps and global_step >= max_steps:
                stop_training = True
                break
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_size_current = inputs.size(0)
                running_val_loss += loss.item() * batch_size_current
                running_val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_val_samples += batch_size_current
        
        # --- METRICS CALCULATION ---
        avg_train_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0
        avg_train_acc = running_train_correct / total_train_samples if total_train_samples > 0 else 0
        
        avg_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0
        avg_val_acc = running_val_correct / total_val_samples if total_val_samples > 0 else 0
        
        # --- PHYSICS ANALYSIS (Optional) ---
        bch_metric = 0.0
        fisher_metric = 0.0
        if analyze_physics:
            # Note: Fisher Rank calculation is computationally expensive
            bch_metric = Analyzer.calculate_bch_error(model)
            fisher_metric = Analyzer.calculate_fisher_rank(model, test_loader)
            
        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['bch'].append(bch_metric)
        history['fisher_rank'].append(fisher_metric)
        
        # Update Epoch Progress Bar
        epoch_pbar.set_postfix({
            'T_Loss': f"{avg_train_loss:.3f}", 
            'V_Loss': f"{avg_val_loss:.3f}",
            'V_Acc': f"{avg_val_acc:.3f}"
        })
        
        # Log to File
        if config.LOGGER:
            config.LOGGER.info(
                f"Epoch {epoch_idx+1} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {avg_val_acc:.4f}"
            )

    # 5. Finalization
    total_training_time = time.time() - start_time
    history['time'] = total_training_time
    
    # Save Model State
    if save_model_path:
        full_save_path = os.path.join(config.RUN_DIR, save_model_path)
        torch.save(model.state_dict(), full_save_path)
        if config.LOGGER: 
            config.LOGGER.info(f"[PERSISTENCE] Model saved to: {full_save_path}")

    return history, avg_val_acc, total_training_time