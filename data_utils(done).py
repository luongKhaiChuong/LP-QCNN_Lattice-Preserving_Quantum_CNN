import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import sys
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')

def get_dataloaders(
    name: str = "MNIST", 
    classes: Optional[List[int]] = [0, 1], 
    data_dir: str = DEFAULT_INPUT_DIR, 
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        name (str): Dataset name
        classes (list): has two mode ()
        data_dir (str): Đường dẫn folder input (mặc định là folder 'input' cùng cấp file code).
        batch_size (int)
    """
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    logger.info(">>> Creating dataset...")

    transform_list = [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]

    if name == "CIFAR10":
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
        
    transform = transforms.Compose(transform_list)
    
    def load_dataset(is_train=True):
        dataset = None
        split_name = "TRAIN" if is_train else "VAL/TEST"

        log_msg = f"Đang tải {split_name} - {name}"
        if classes is None:
            logger.info(f"{log_msg} (Full)...")
        else:
            logger.info(f"{log_msg} (Class {classes})...")

        try:
            if name == "MNIST":
                dataset = datasets.MNIST(root=data_dir, train=is_train, download=True, transform=transform)
            elif name == "FashionMNIST":
                dataset = datasets.FashionMNIST(root=data_dir, train=is_train, download=True, transform=transform)
            elif name == "CIFAR10":
                dataset = datasets.CIFAR10(root=data_dir, train=is_train, download=True, transform=transform)
                dataset.targets = torch.tensor(dataset.targets)
            else:
                raise ValueError(f"Dataset {name} not supported.")
                
        except Exception as e:
            logger.error(f"Error when downloading {name}: {e}")
            raise e

        if hasattr(dataset, 'labels'):
            dataset.targets = torch.tensor(dataset.labels.squeeze()).long()
            if hasattr(dataset, 'imgs'):
                dataset.data = dataset.imgs

        if not isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(dataset.targets)

        if classes is not None:
            idx = (dataset.targets == classes[0]) | (dataset.targets == classes[1])
            dataset.targets = dataset.targets[idx]
            
            if isinstance(dataset.data, np.ndarray):
                dataset.data = dataset.data[idx.numpy() if isinstance(idx, torch.Tensor) else idx]
            elif isinstance(dataset.data, torch.Tensor):
                dataset.data = dataset.data[idx]
            else:
                dataset.data = dataset.data[idx]
            
            dataset.targets = torch.where(dataset.targets == classes[0], 0, 1)
        
        return dataset

    train_subset = load_dataset(is_train=True)
    val_subset = load_dataset(is_train=False)

    logger.info(f">>> Loaded successfully. Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader

if __name__ == "__main__":
  
    try:
        train_loader, val_loader = get_dataloaders(name="MNIST", classes=None, batch_size=64)
        
        batch = next(iter(train_loader))
        logger.info(f">>> Test batch shape: {batch[0].shape}")
        
    except Exception as e:
        logger.critical(f">>> Error: {e}")