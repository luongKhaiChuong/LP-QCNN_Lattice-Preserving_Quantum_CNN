import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import config

def get_dataloaders(dataset_name, batch_size, classes=None, grayscale=True):
    data_path = os.path.join(config.INPUT_DIR, dataset_name)
    os.makedirs(data_path, exist_ok=True)
    
    # Transforms
    transform_list = [
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ]
    
    if dataset_name == "CIFAR10":
        if grayscale:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            
    transform = transforms.Compose(transform_list)
    print(f"Loading {dataset_name} | Batch: {batch_size} | Mode: {'Grayscale' if grayscale else 'RGB'}")

    # Load Dataset Raw
    if dataset_name == "MNIST":
        train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        train_data = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
        val_data = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        val_data = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Filter Classes (Optional)
    if classes is not None:
        def filter_subset(dataset):
            if not isinstance(dataset.targets, torch.Tensor):
                targets = torch.tensor(dataset.targets)
            else:
                targets = dataset.targets
            idx = (targets == classes[0]) | (targets == classes[1])
            dataset.targets = targets[idx]
            dataset.data = dataset.data[idx]
            dataset.targets = torch.where(dataset.targets == classes[0], 0, 1)
            return dataset

        train_data = filter_subset(train_data)
        val_data = filter_subset(val_data)

    # DEBUG TRUNCATION (100/20)
    if config.DEBUG:
        print(f">>> DEBUG: Taking subset (Train: {config.DEBUG_TRAIN_SIZE}, Val: {config.DEBUG_VAL_SIZE})")
        t_size = min(len(train_data), config.DEBUG_TRAIN_SIZE)
        v_size = min(len(val_data), config.DEBUG_VAL_SIZE)
        
        train_data = Subset(train_data, range(t_size))
        val_data = Subset(val_data, range(v_size))

    # Dataloaders
    num_workers = 0 if config.DEBUG else 2
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return train_loader, val_loader