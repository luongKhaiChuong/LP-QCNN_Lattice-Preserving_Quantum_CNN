import torch
import torch.nn as nn
import time
import os
import config

def train_engine(model, train_loader, val_loader, save_path, mode='epoch', max_duration=10):
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_duration * 2)
    
    history = {'step': [], 'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_acc = 0.0
    
    # --- HELPER FUNCTIONS ---
    def run_val():
        model.eval()
        correct, total = 0, 0
        running_loss = 0.0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(config.DEVICE), lbl.to(config.DEVICE)
                out = model(img)
                loss = loss_fn(out, lbl)
                running_loss += loss.item() * img.size(0)
                correct += (out.argmax(1) == lbl).sum().item()
                total += lbl.size(0)
        acc = correct / total if total > 0 else 0
        avg_loss = running_loss / total if total > 0 else 0
        return acc, avg_loss

    def train_step(img, lbl):
        img, lbl = img.to(config.DEVICE), lbl.to(config.DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = loss_fn(out, lbl)
        loss.backward()
        optimizer.step()
        acc = (out.argmax(1) == lbl).float().mean().item()
        return loss.item(), acc
    
    model.eval()
    try:
        init_img, init_lbl = next(iter(train_loader))
        init_img, init_lbl = init_img.to(config.DEVICE), init_lbl.to(config.DEVICE)
        with torch.no_grad():
            init_out = model(init_img)
            init_loss = loss_fn(init_out, init_lbl).item()
            init_acc = (init_out.argmax(1) == init_lbl).float().mean().item()
    except StopIteration:
        init_loss, init_acc = 0.0, 0.0
    
    init_val_acc, init_val_loss = run_val()
    
    if mode == 'iter':
        history['step'].append(0)
        history['train_acc'].append(init_acc)
        history['train_loss'].append(init_loss)
        history['val_acc'].append(init_val_acc)
        history['val_loss'].append(init_val_loss)

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    print(f"   -> Start Training [{mode.upper()}] (Total: {max_duration})")
    
    if mode == 'iter':
        model.train()
        data_iter = iter(train_loader)
        
        for it in range(max_duration):
            try:
                img, lbl = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                img, lbl = next(data_iter)
            
            loss, acc = train_step(img, lbl)
            scheduler.step() 
            
            if (it + 1) % 20 == 0 or (it + 1) == max_duration:
                val_acc, val_loss = run_val()
                
                history['step'].append(it + 1) # 20, 40, 60...
                history['train_acc'].append(acc)
                history['train_loss'].append(loss)
                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)
                
                model.train() 
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), save_path)

    else: # Epoch Mode
        for epoch in range(max_duration):
            model.train()
            running_correct, running_loss, total_samples = 0, 0, 0
            
            for img, lbl in train_loader:
                loss, acc = train_step(img, lbl)
                running_correct += acc * img.size(0)
                running_loss += loss * img.size(0)
                total_samples += img.size(0)
            
            scheduler.step()
            
            train_acc = running_correct / total_samples
            train_loss = running_loss / total_samples
            val_acc, val_loss = run_val()
            
            history['step'].append(epoch + 1) # 1, 2, 3...
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
    
    return history