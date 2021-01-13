# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

# %%
# Parameters
DATASET_PATH = "data"  # path to extracted http://awe.fri.uni-lj.si/downloads/AWEDataset.zip
BATCH_SIZE = 10
SHUFFLE_SIZE = 500
AWE_W = 480
AWE_H = 352  # divisible by 32
AWE_C = 3
GROUP_NORM = 16
EPOCHS = 35
EXP_ID = "initial"  # subfolder inside `out/` with saved weights
TRAIN = False  # `True` = train, `False` = load saved checkpoints
OUT_DIR = os.path.join("out", EXP_ID)

# %%
# Load image paths.
translations = pd.read_csv('awe-translation.csv')
images = dict()
labels = dict()
for dataset in ['train', 'test']:
    rows = translations['AWE-Full image path'].str.startswith(dataset)
    images[dataset] = list(translations[rows]['AWE image path'])
    labels[dataset] = list(translations[rows]['Subject ID'])

# %%
