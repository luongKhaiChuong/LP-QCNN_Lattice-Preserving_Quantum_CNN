import torch
import torch.nn as nn
from tqdm.auto import tqdm
import time
import config

def train_model(model, train_loader, val_loader, save_path="best_model.pth"):
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"\n>>> Start training on {config.DEVICE}...")
    start_total = time.time()

    def run_validation():
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(config.DEVICE), label.to(config.DEVICE)
                out = model(img)
                val_loss_sum += loss_fn(out, label).item() * img.size(0)
                val_correct += (out.argmax(1) == label).sum().item()
                val_total += label.size(0)
        return val_loss_sum / val_total, val_correct / val_total

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for img, label in pbar:
            img, label = img.to(config.DEVICE), label.to(config.DEVICE)
            
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img.size(0)
            running_correct += (out.argmax(1) == label).sum().item()
            running_total += label.size(0)
            
            curr_acc = running_correct / running_total
            pbar.set_postfix({'acc': f'{curr_acc:.4f}'})

        epoch_train_loss = running_loss / running_total
        epoch_train_acc = running_correct / running_total
        
        val_loss, val_acc = run_validation()
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc {epoch_train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model: {best_val_acc*100:.2f}%")

    print(f"\nTotal time: {time.time() - start_total:.1f}s")