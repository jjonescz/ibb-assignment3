# %%
import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

# %%
# Initialize.
FIGURES_DIR = 'figures'

os.makedirs(FIGURES_DIR, exist_ok=True)

# %%
# Load train history.
with open('out/23-final/train_history.pkl', 'rb') as f:
    train_historyA = pickle.load(f)
with open('out/24-augmentations/train_history.pkl', 'rb') as f:
    train_historyB = pickle.load(f)

# %%
# Plot train evolution.
plt.plot(train_historyA['accuracy'], 'r', label='training without aug.')
plt.plot(train_historyA['val_accuracy'], 'r--', label='validation without aug.')
plt.plot(train_historyB['accuracy'], 'g', label='training with aug.')
plt.plot(train_historyB['val_accuracy'], 'g--', label='validation with aug.')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(os.path.join(FIGURES_DIR, 'acc.pdf'), bbox_inches='tight', pad_inches=0)
plt.plot()

# %%
# Load CMC.
with open('out/23-final/cmc.pkl', 'rb') as f:
    cmcA = pickle.load(f)
with open('out/24-augmentations/cmc.pkl', 'rb') as f:
    cmcB = pickle.load(f)

# %%
# Plot CMC curves.
plt.plot(range(1, len(cmcA) + 1), cmcA, label='without augmentations')
plt.plot(range(1, len(cmcB) + 1), cmcB, label='with augmentations')
plt.xticks([1] + list(range(20, len(cmcA) + 1, 20)))
plt.ylabel('recognition rate')
plt.xlabel('rank')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'cmc.pdf'), bbox_inches='tight', pad_inches=0)
plt.show()

# %%
# Report performance metrics.
print(f'Without augmentations: Rank-1={cmcA[0] * 100:.2f}, Rank-5={cmcA[4] * 100:.2f}, AUCMC={np.trapz(cmcA):.2f}')
print(f'With augmentations: Rank-1={cmcB[0] * 100:.2f}, Rank-5={cmcB[4] * 100:.2f}, AUCMC={np.trapz(cmcB):.2f}')

# %%
