# IBB Assignment 3

This repository contains source code of my solution of Assignment 3 for Image
Based Biometry course at University of Ljubljana.

Report with IMRAD structure is available as [release asset](https://github.com/jjonescz/ibb-assignment3/releases).

## Requirements

Python 3.8.2 was used with the following packages installed:

```txt
matplotlib==3.3.2
numpy==1.18.5
pandas==1.1.3
tensorflow==2.3.1
```

Additionally, folder `data` must contain [AWE dataset](http://awe.fri.uni-lj.si/downloads/AWEDataset.zip) unzipped so that e.g. `data/001/01.png` is a valid path.

## Loading and training

Script `train.py` can be run as-is.
It will download EfficientNet-B0 weights and load saved weights of our CNN model trained without image augmentations.
To switch to model with image augmentations, change parameters near top of the file to contain:

```py
EXP_ID = "model-b"
AUGMENTATIONS = True
```

To enable model training, change parameters to:

```py
TRAIN = True
```

## Evaluation

Script `evaluate.py` plots figures (to folder `figures`) and prints performance metrics to console.
It uses state of models provided in folder `out`.
This state can be recomputed by executing script `train.py` (once with augmentations and once without them) as described in previous section.

## Report

Source code of LaTeX report is contained in `report/jj1712.tex`.
Before compiling it, make sure you have generated figures as described in [Evaluation](#evaluation).
