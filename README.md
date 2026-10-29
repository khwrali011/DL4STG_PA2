# DL4STG PA2 — Question 2

Moving MNIST forecasting with ConvLSTM: baseline A, symmetry augmentation B, speed augmentation C, and dilated model D. Each model was trained for 30 epochs with seeds 0 and 1 on Kaggle Tesla T4 GPUs. Symmetrised B and D were also evaluated.

## Files

- `Kaggle_Run.ipynb`: experiment runner.
- `*.py`: model, training, evaluation, plotting, and reporting code.
- `outputs/`: metrics, Tables 1–6, figures, and eight GIFs.
- `predictions/`: Q2.5 and Q2.11 predictions, recorded in the original Git history.
- `requirements.txt`: dependencies.

## Run

Attach the source package to a Kaggle notebook, enable GPU T4 ×2 and Internet, and run `Kaggle_Run.ipynb` in order: setup → baseline → variants → final. Review the prediction text before each experiment. Keep Kaggle's compatible PyTorch and torchvision versions.

Raw data and checkpoints are excluded from Git. Setup downloads the data; reproducing evaluation without checkpoints requires training. This repository covers Q2; the combined Q1+Q2 report is submitted separately.
