import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import config
import os

def aux_download_data():
    print(f"[DATA] Checking/Downloading datasets to {config.DATA_ROOT}...")

    datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
    datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
    datasets.FashionMNIST(root=config.DATA_ROOT, train=True, download=True)
    datasets.FashionMNIST(root=config.DATA_ROOT, train=False, download=True)
    
    print("[DATA] All datasets are ready.")

def get_dataloaders(dataset_name, data_size, batch_size, binary_mode=False):
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor()
    ])
    
    ds_class = getattr(datasets, dataset_name)
    train_full = ds_class(root=config.DATA_ROOT, train=True, download=True, transform=transform)
    test_full = ds_class(root=config.DATA_ROOT, train=False, download=True, transform=transform)

    if binary_mode:
        idx_train = (train_full.targets == 0) | (train_full.targets == 1)
        idx_test = (test_full.targets == 0) | (test_full.targets == 1)
        
        train_full.data = train_full.data[idx_train]
        train_full.targets = train_full.targets[idx_train]
        test_full.data = test_full.data[idx_test]
        test_full.targets = test_full.targets[idx_test]

    if data_size != 'fullset':
        real_len = len(train_full)
        use_size = min(data_size, real_len)
        
        g = torch.Generator()
        g.manual_seed(config.SEED)
        indices = torch.randperm(real_len, generator=g)[:use_size]
        
        train_set = Subset(train_full, indices)
        
        test_len = len(test_full)
        test_size = min(test_len, max(100, int(use_size * 0.2)))
        test_indices = torch.randperm(test_len, generator=g)[:test_size]
        test_set = Subset(test_full, test_indices)
    else:
        train_set = train_full
        test_set = test_full

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader