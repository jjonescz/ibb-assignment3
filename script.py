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
IMAGE_W = 128
IMAGE_H = 128
IMAGE_C = 3
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
    filenames = translations[rows]['AWE image path']
    paths = map(lambda f: os.path.join(DATASET_PATH, f), filenames)
    images[dataset] = list(paths)
    labels[dataset] = list(translations[rows]['Subject ID'])

# %%
# Load images as Tensorflow Datasets.
datasets = dict()
for dataset in ['train', 'test']:
    def transform(image, label):
        # Load image from given path.
        image = tf.io.read_file(image)
        image = tf.io.decode_png(image, channels=IMAGE_C)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((images[dataset], labels[dataset]))
    datasets[dataset] = ds.map(transform)

# %%
# Determine image shapes.
image_shapes = np.array([image.shape for image, _ in datasets['train']])
print(f'Min image size: {image_shapes.min(axis=0)}')
print(f'Max image size: {image_shapes.max(axis=0)}')
print(f'Avg image size: {image_shapes.mean(axis=0)}')

# %%
# Resize images.
for dataset, ds in datasets.items():
    def transform(image, label):
        image = tf.image.resize(image, (IMAGE_H, IMAGE_W))
        return image, label
    datasets[dataset] = ds.map(transform)

# %%
# Split into training, validation and testing datasets.
ds_train = datasets['train'].take(500).cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
ds_val = datasets['train'].skip(500).cache().batch(BATCH_SIZE)
ds_test = datasets['test'].cache().batch(BATCH_SIZE)

# %%
