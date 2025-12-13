import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WIRES = 4
N_FILTERS = 6
TIME = 5
TIME_STEPS = 5
KERNEL_SIZE = 2
STRIDE = 2
FIN_KERNEL = (7, 7)

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
ITERATIONS = 200

CHECKPOINT_DIR = "./checkpoints"