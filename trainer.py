import torch
import torch.nn as nn
import time
import config
import os
from tqdm import tqdm
from utils import seed_everything
from data import get_dataloaders
from model import LPQCNN
from analyzer import Analyzer

def train_experiment(config_dict, run_name, analyze_physics=False, max_steps=None, save_model_path=None):
    seed_everything(config.SEED)
    binary = config_dict.get('binary_mode', False)
    tr_load, te_load = get_dataloaders(config_dict['dataset'], config_dict['data_size'], 
                                     batch_size=128 if 'batch_size' not in config_dict else config_dict['batch_size'],
                                     binary_mode=binary)
    
    n_classes = 2 if binary else 10
    model = LPQCNN(n_filters=config_dict['n_filters'], time_dyn=config_dict['time'], 
                   time_steps=config_dict['time_steps'], n_classes=n_classes).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'bch': [], 'fisher_rank': [], 
        'time': 0, 'config': config_dict
    }
    
    if config.LOGGER:
        config.LOGGER.info(f">>> RUN: {run_name} | {config_dict}")
    
    global_step = 0
    stop_training = False
    start_train_time = time.time()
    
    epoch_pbar = tqdm(range(config_dict['epochs']), desc=f"Training {run_name}", position=0, leave=True)
    
    for ep in epoch_pbar:
        if stop_training: break
        t0 = time.time()
        model.train()
        train_loss = 0; train_correct = 0; total_train = 0
        
        batch_pbar = tqdm(tr_load, desc=f"Ep {ep+1}", leave=False, position=1)
        
        for x, y in batch_pbar:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            total_train += x.size(0)
            
            batch_pbar.set_postfix({'loss': loss.item()})
            
            global_step += 1
            if max_steps and global_step >= max_steps:
                stop_training = True
                break
        
        model.eval()
        val_loss = 0; val_acc = 0; val_total = 0
        with torch.no_grad():
            for x, y in te_load:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item() * x.size(0)
                val_acc += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)
        
        avg_tr_loss = train_loss/total_train if total_train>0 else 0
        avg_tr_acc = train_correct/total_train if total_train>0 else 0
        avg_val_loss = val_loss/val_total if val_total>0 else 0
        avg_val_acc = val_acc/val_total if val_total>0 else 0
        
        bch_val, fisher_val = 0.0, 0.0
        if analyze_physics:
            bch_val = Analyzer.calculate_bch_error(model)
            fisher_val = Analyzer.calculate_fisher_rank(model, te_load)
            
        history['train_loss'].append(avg_tr_loss); history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_tr_acc); history['val_acc'].append(avg_val_acc)
        history['bch'].append(bch_val); history['fisher_rank'].append(fisher_val)
        
        epoch_pbar.set_postfix({
            'TL': f"{avg_tr_loss:.3f}", 
            'VL': f"{avg_val_loss:.3f}",
            'VA': f"{avg_val_acc:.3f}"
        })
        
        if config.LOGGER:
            config.LOGGER.info(f"Ep {ep+1} | T.Loss: {avg_tr_loss:.4f} V.Loss: {avg_val_loss:.4f} | V.Acc: {avg_val_acc:.4f}")

    total_time = time.time() - start_train_time
    history['time'] = total_time
    
    if save_model_path:
        full_save_path = os.path.join(config.RUN_DIR, save_model_path)
        torch.save(model.state_dict(), full_save_path)
        if config.LOGGER: config.LOGGER.info(f"[SAVED] Model: {full_save_path}")

    return history, avg_val_acc, total_time