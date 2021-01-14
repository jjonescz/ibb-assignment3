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

Additionally, `data/` folder must contain [AWE dataset](http://awe.fri.uni-lj.si/downloads/AWEDataset.zip) unzipped so that e.g. `data/001/01.png` is a valid path.

## Training

Script `train.py` can be run as-is.
It will download EfficientNet-B0 weights and train our CNN model without image augmentations.
To enable image augmentations, change variables near top of the file to contain:

```py
EXP_ID = "model-b"
AUGMENTATIONS = True
```

## Evaluation

Results needed to evaluate models trained without and with image augmentations are provided in folders `out/model-a` and `out/model-b`, respectively.
To evaluate these two models, run script `evaluate.py`.
Results will be printed to console and figures saved to folder `figures`.

## Report

Source code of LaTeX report is contained in `report/jj1712.tex`.
Before compiling it, make sure you have generated figures as described in [Evaluation](#evaluation).
