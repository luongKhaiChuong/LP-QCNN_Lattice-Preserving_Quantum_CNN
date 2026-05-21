import torch
import numpy as np
import random
import json
import config
import os
import logging
import sys
from datetime import datetime

def seed_everything(random_seed=42):
    """
    Sets the random seed across all libraries (Python, NumPy, PyTorch) to ensure 
    experimental reproducibility.

    Args:
        random_seed (int): The seed value to initialize random number generators. 
                           Default is 42.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        # Force cuDNN to use deterministic algorithms (may slow down performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_run_dir():
    """
    Creates a unique timestamped directory for the current experimental run.
    
    The directory format is: 'outputs/YYYY-MM-DD_HH-MM-SS_MODE'
    This ensures that results from different runs do not overwrite each other.
    """
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder_name = f"{current_timestamp}_{config.MODE}"
    
    # Update the global configuration with the new path
    config.RUN_DIR = os.path.join(config.OUTPUT_BASE, run_folder_name)
    
    if not os.path.exists(config.RUN_DIR):
        os.makedirs(config.RUN_DIR)
        
    print(f"\n[SYSTEM] Output Directory Created: {config.RUN_DIR}")

def setup_logger():
    """
    Initializes the system logger to track experiment progress.
    
    This function configures two handlers:
    1. FileHandler: Saves all logs to 'execution.log' inside the run directory.
    2. StreamHandler: Prints logs to the console (stdout).
    """
    log_file_path = os.path.join(config.RUN_DIR, "execution.log")
    
    logger = logging.getLogger("Experiment")
    logger.setLevel(logging.INFO)
    
    # Define log format: Timestamp | Level | Message
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 1. File Handler Setup
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. Stream Handler Setup (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Assign to global config for access across modules
    config.LOGGER = logger
    logger.info(f"Logger initialized. Saving logs to: {log_file_path}")

def save_to_json(data_dict, filename="final_results"):
    """
    Serializes and saves experimental data to a JSON file.

    This function handles non-serializable types common in ML (e.g., numpy ints/floats, 
    PyTorch tensors) by converting them to standard Python types.

    Args:
        data_dict (dict): The dictionary containing results or metrics to save.
        filename (str): The base name of the output file (excluding extension).
    """
    def json_serializer(obj):
        """Helper to convert objects to JSON-serializable formats."""
        if isinstance(obj, (np.int64, np.int32)): 
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)): 
            return float(obj)
        if isinstance(obj, torch.Tensor): 
            return obj.item()
        return obj
    
    # Construct full path: outputs/.../filename_MODE.json
    final_filename = f"{filename}_{config.MODE}.json"
    save_path = os.path.join(config.RUN_DIR, final_filename)
    
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, default=json_serializer, indent=4)
    
    if config.LOGGER:
        config.LOGGER.info(f"[PERSISTENCE] Full data saved to {save_path}")
    else:
        print(f"[INFO] Full data saved to {save_path}")

def generate_grid_topology(height, width):
    """
    Generates the connectivity graph for a 2D grid lattice.
    
    This function maps the 2D grid coordinates to 1D qubit indices and identifies 
    all adjacent pairs for Horizontal and Vertical interactions.

    Args:
        height (int): The number of rows in the grid (GRID_H).
        width (int): The number of columns in the grid (GRID_W).

    Returns:
        tuple: (horizontal_pairs, vertical_pairs)
            - horizontal_pairs (list of lists): Connections between (row, col) and (row, col+1).
            - vertical_pairs (list of lists): Connections between (row, col) and (row+1, col).
    """
    horizontal_pairs = []
    vertical_pairs = []
    
    # 1. Generate Horizontal Edges (East-West connections)
    for row in range(height):
        for col in range(width - 1):
            node_u = row * width + col
            node_v = row * width + col + 1
            horizontal_pairs.append([node_u, node_v])
            
    # 2. Generate Vertical Edges (North-South connections)
    for row in range(height - 1):
        for col in range(width):
            node_u = row * width + col
            node_v = (row + 1) * width + col
            vertical_pairs.append([node_u, node_v])
            
    return horizontal_pairs, vertical_pairs