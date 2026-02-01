import torch


MODE = 'DUMMY' #DUMMY for functionality test, FULL for the whole run
SEED = 42

# Size of the grid
GRID_H = 2
GRID_W = 2
N_WIRES = GRID_H * GRID_W

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Result space for saving
GLOBAL_RESULTS = {}

DATA_ROOT = './data'
OUTPUT_BASE = './outputs'
RUN_DIR = None
LOGGER = None