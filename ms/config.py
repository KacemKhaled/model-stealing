import os
import os.path as osp
from os.path import dirname, abspath
import torch
import socket

host_name = socket.gethostname()

DEFAULT_SEED = 42
DS_SEED = 123  # uses this seed when splitting datasets

if host_name == 'heslab-srv':
    CC = False  # No Compute Canada
    KK = False
    print("INFO: Using HESLab Server")
elif host_name == 'LAPTOP-D1BTJ0C3':
    CC = False
    KK = True
    print("INFO: Using local laptop")
else:
    CC = True  # Compute Canada
    KK = False
    print("INFO: Using Compute Canada Server")

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
CACHE_ROOT = osp.join(SRC_ROOT, 'cache')
DATASET_ROOT = osp.join(PROJECT_ROOT, 'data')
# specific to server related dataset folder
# DATASET_ROOT = osp.join(os.sep, 'homeHDD', 'kacem', 'data')
ROOT_SSD_Data = osp.join(os.sep, 'SSD_Data', 'kacem', 'model_stealing')
DATASET_ROOT_SDD = osp.join(ROOT_SSD_Data, 'data')
MODEL_DIR_SSD = osp.join(ROOT_SSD_Data, 'models')

# specific to compute canada node server related dataset folder
DATASET_ROOT_CC = '/home/kacemkh/scratch/kacem/datasets/'  # osp.join(os.sep, 'homeHDD', 'kacem', 'data')

DEBUG_ROOT = osp.join(PROJECT_ROOT, 'debug')
MODEL_DIR = osp.join(PROJECT_ROOT, 'models')

DATASET_ROOT = DATASET_ROOT_CC if CC else DATASET_ROOT
if not KK:
    if osp.exists(DATASET_ROOT_SDD):
        print(f"INFO: Using external path for data: '{DATASET_ROOT_SDD}' instead of '{DATASET_ROOT}'")
        DATASET_ROOT = DATASET_ROOT_SDD

    if osp.exists(MODEL_DIR_SSD):
        print(f"INFO: Using external path for models: '{MODEL_DIR_SSD}' instead of '{MODEL_DIR}'")
        MODEL_DIR = MODEL_DIR_SSD

# -------------- WANDB Stuff
WB_PROJECT = "MSF"
WB_ENTITY = "anonymous_submission" if CC else "kacem"

# -------------- Dataset Stuff
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_BATCH_SIZE = 64

# --
AVAIL_GPUS = torch.cuda.device_count() #-1 if CC else torch.cuda.device_count()
print('INFO AVAIL GPUS : ', AVAIL_GPUS)
# NUM_NODES = int(os.environ.get("SLURM_JOB_NUM_NODES")) if CC else 1
REFRESH_RATE = 100  # if CC else 10
ACCELERATOR = 'ddp' if CC else 'dp'
STRATEGY = 'ddp' if CC else 'dp'
