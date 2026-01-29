import torch
import numpy as np
import random
import json
import config
import os
import logging
import sys
from datetime import datetime

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_run_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{config.MODE}"
    config.RUN_DIR = os.path.join(config.OUTPUT_BASE, run_name)
    
    if not os.path.exists(config.RUN_DIR):
        os.makedirs(config.RUN_DIR)
        
    print(f"\n[SYSTEM] Output Directory Created: {config.RUN_DIR}")

def setup_logger():
    """Thiết lập logger để ghi ra file và màn hình"""
    log_file = os.path.join(config.RUN_DIR, "execution.log")
    
    logger = logging.getLogger("Experiment")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    config.LOGGER = logger
    logger.info(f"Logger initialized. Saving logs to: {log_file}")

def save_to_json(data, filename="final_results.json"):
    def convert(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32): return int(o)
        if isinstance(o, np.float32) or isinstance(o, np.float64): return float(o)
        if isinstance(o, torch.Tensor): return o.item()
        return o
    
    final_name = f"{filename.split('.')[0]}_{config.MODE}.json"
    save_path = os.path.join(config.RUN_DIR, final_name)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, default=convert, indent=4)
    
    if config.LOGGER:
        config.LOGGER.info(f"FULL DATA saved to {save_path}")
    else:
        print(f"[INFO] FULL DATA saved to {save_path}")

def generate_grid_topology(h, w):
    h_pairs = []
    v_pairs = []
    for r in range(h):
        for c in range(w - 1):
            u = r * w + c
            v = r * w + c + 1
            h_pairs.append([u, v])
    for r in range(h - 1):
        for c in range(w):
            u = r * w + c
            v = (r + 1) * w + c
            v_pairs.append([u, v])
    return h_pairs, v_pairs