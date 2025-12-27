import torch

DEBUG = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- EXPERIMENT SETTINGS ---
NUM_TRIALS = 10 

NUM_WIRES = 4 
DEFAULT_NUM_FILTERS = 6 

FILTER_RANGE_GRAYSCALE = list(range(1, 13))
FILTER_RANGE_RGB = list(range(3, 16)) 

# --- DYNAMICS ---
TIME_DURATION = 5
TIME_STEPS = 5

# --- CONVOLUTION ---
STRIDE = 1 

# --- TRAINING CONFIG ---
LEARNING_RATE = 0.01
BATCH_SIZE_EPOCH = 128
BATCH_SIZE_ITER = 25

MAX_EPOCHS = 10
MAX_ITERS = 200

# Debug Config
DEBUG_TRAIN_SIZE = 100
DEBUG_VAL_SIZE = 20
DEBUG_BATCH_SIZE = 1

# --- PATHS ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"

def set_debug_mode(is_debug: bool):
    global DEBUG, NUM_WIRES, MAX_EPOCHS, MAX_ITERS, NUM_TRIALS
    global BATCH_SIZE_EPOCH, BATCH_SIZE_ITER
    global FILTER_RANGE_GRAYSCALE, FILTER_RANGE_RGB
    
    DEBUG = is_debug
    
    if DEBUG:
        print(f"\n{'!'*50}")
        print(f"!!! DEBUG MODE ACTIVATED !!!")
        print(f"!!! Data: {DEBUG_TRAIN_SIZE}/{DEBUG_VAL_SIZE}, Batch: {DEBUG_BATCH_SIZE}")
        print(f"{'!'*50}\n")
        
        NUM_WIRES = 1
        BATCH_SIZE_EPOCH = DEBUG_BATCH_SIZE
        BATCH_SIZE_ITER = DEBUG_BATCH_SIZE

        MAX_EPOCHS = 3
        MAX_ITERS = 100
        NUM_TRIALS = 3
        
        FILTER_RANGE_GRAYSCALE = [1, 2, 3]
        FILTER_RANGE_RGB = [3, 6, 9]
    else:
        NUM_WIRES = 4
        BATCH_SIZE_EPOCH = 128
        BATCH_SIZE_ITER = 25
        
        MAX_EPOCHS = 10
        MAX_ITERS = 200
        NUM_TRIALS = 10 
        
        FILTER_RANGE_GRAYSCALE = list(range(1, 13))
        FILTER_RANGE_RGB = list(range(3, 16))