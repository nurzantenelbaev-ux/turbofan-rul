"""Global configuration: seeds, paths, sensor lists, hardware settings."""
import os
import random
import multiprocessing as mp

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# -------------------------------------------------------------------
# Hardware detection
# -------------------------------------------------------------------
N_CPUS = mp.cpu_count()          # used for parallel feature extraction / RF
N_JOBS = max(1, N_CPUS - 1)      # leave one core free for the OS

_HAS_TORCH = False
_HAS_CUDA = False
try:
    import torch
    _HAS_TORCH = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

DEVICE = "cuda" if _HAS_CUDA else "cpu"


def print_hardware():
    """Call at the start of a run so you can see what it's using."""
    print("=" * 50)
    print("HARDWARE")
    print("=" * 50)
    print(f"CPU cores detected : {N_CPUS} (using {N_JOBS} workers)")
    if _HAS_TORCH and _HAS_CUDA:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU detected       : {name} ({vram_gb:.1f} GB VRAM)")
        print(f"PyTorch device     : cuda")
    elif _HAS_TORCH:
        print("GPU detected       : none — falling back to CPU for PyTorch")
    else:
        print("PyTorch            : not installed")
    print("=" * 50)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# -------------------------------------------------------------------
# Column names for the C-MAPSS .txt files
# -------------------------------------------------------------------
COLS = (["unit_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)])

ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]

# Sensors that are constant or near-constant in FD001 (verify in your EDA).
CONSTANT_SENSORS_FD001 = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

# -------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------
WINDOW_SIZE = 30
STRIDE = 1
R_EARLY = 125           # piecewise linear RUL ceiling
N_BASELINE_CYCLES = 30  # first N cycles used as the "healthy baseline"

# LSTM batch size — raise for GPU (fits easily in 4 GB on 1050 Ti), lower for CPU.
LSTM_BATCH = 512 if _HAS_CUDA else 128
