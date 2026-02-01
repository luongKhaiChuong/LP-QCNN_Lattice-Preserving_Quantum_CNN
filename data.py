import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import config
import os

def aux_download_datasets():
    """
    Downloads the MNIST and FashionMNIST datasets to the configured data directory.
    
    This function is intended to be run once before experiments begin, especially 
    in HPCC environments where compute nodes might not have internet access.
    It checks the `config.DATA_ROOT` directory.
    """
    print(f"[DATA] Checking/Downloading datasets to {config.DATA_ROOT}...")

    # Download MNIST (Train & Test)
    datasets.MNIST(root=config.DATA_ROOT, train=True, download=True)
    datasets.MNIST(root=config.DATA_ROOT, train=False, download=True)
    
    # Download FashionMNIST (Train & Test)
    datasets.FashionMNIST(root=config.DATA_ROOT, train=True, download=True)
    datasets.FashionMNIST(root=config.DATA_ROOT, train=False, download=True)
    
    print("[DATA] All datasets are ready.")

def get_dataloaders(dataset_name, data_size, batch_size, binary_mode=False):
    """
    Creates and returns PyTorch DataLoaders for training and testing.

    ### Process:
    1. Applies image transformations: Resize to 14x14 (to fit quantum circuit width) and convert to Tensor.
    2. Loads the specified dataset (MNIST or FashionMNIST).
    3. (Optional) Filters data for binary classification (Classes 0 and 1 only).
    4. Subsets the data if `data_size` is specified (e.g., for quick testing or data efficiency analysis).
    5. Wraps the datasets in DataLoaders for batched iteration.

    Args:
        dataset_name (str): The name of the dataset ('MNIST' or 'FashionMNIST').
        data_size (int or str): The number of training samples to use. 
                                Pass 'fullset' to use the entire dataset.
        batch_size (int): The number of samples per batch during training/testing.
        binary_mode (bool): If True, filters the dataset to only include class 0 and class 1.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define preprocessing pipeline: Resize -> ToTensor
    # Resizing to 14x14 is crucial for LP-QCNN to map to the qubit grid size.
    transform_pipeline = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor()
    ])
    
    # Dynamically load the dataset class from torchvision
    dataset_class = getattr(datasets, dataset_name)
    
    train_full_dataset = dataset_class(root=config.DATA_ROOT, train=True, download=True, transform=transform_pipeline)
    test_full_dataset = dataset_class(root=config.DATA_ROOT, train=False, download=True, transform=transform_pipeline)

    # --- Binary Mode Filtering ---
    if binary_mode:
        # Create boolean masks for classes 0 and 1
        train_mask = (train_full_dataset.targets == 0) | (train_full_dataset.targets == 1)
        test_mask = (test_full_dataset.targets == 0) | (test_full_dataset.targets == 1)
        
        # Apply masks to data and targets
        train_full_dataset.data = train_full_dataset.data[train_mask]
        train_full_dataset.targets = train_full_dataset.targets[train_mask]
        test_full_dataset.data = test_full_dataset.data[test_mask]
        test_full_dataset.targets = test_full_dataset.targets[test_mask]

    # --- Data Subsetting ---
    if data_size != 'fullset':
        total_train_len = len(train_full_dataset)
        subset_train_size = min(data_size, total_train_len)
        
        # Use a deterministic generator for reproducible subsets
        rng = torch.Generator()
        rng.manual_seed(config.SEED)
        
        # Randomly select indices for the training subset
        train_indices = torch.randperm(total_train_len, generator=rng)[:subset_train_size]
        train_subset = Subset(train_full_dataset, train_indices)
        
        # Create a correspondingly smaller test set (20% of train size, min 100)
        total_test_len = len(test_full_dataset)
        subset_test_size = min(total_test_len, max(100, int(subset_train_size * 0.2)))
        test_indices = torch.randperm(total_test_len, generator=rng)[:subset_test_size]
        test_subset = Subset(test_full_dataset, test_indices)
    else:
        # Use the complete datasets
        train_subset = train_full_dataset
        test_subset = test_full_dataset

    # --- DataLoader Creation ---
    # num_workers=0 is safer for compatibility, but can be increased for speed on supported systems
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader