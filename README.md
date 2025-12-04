# 🛒 Customer Re-Purchase / Churn Prediction

This repository contains notebooks and supporting code to predict whether a customer will repurchase (or churn) in the next month. The notebooks cover data cleaning, feature engineering, model training, evaluation and a small PyTorch model implementation.

---

## 📁 Real project structure

Below is the actual repository layout (root = project folder):

```
Notebook_data_cleaning.ipynb        # Data cleaning & feature engineering notebook
Notebook_prediction.ipynb          # Model training, evaluation & experiments
README.md                          # This documentation file

models/                             # Model code and checkpoints
  ├─ __init__.py                    # package init
  ├─ MLP.py                         # PyTorch MLP model implementation
  ├─ model.py                       # higher-level model utilities / wrappers
  └─ checkpoint/
      └─ best_model.pt              # saved model weights (binary)

utils/                              # helper utilities
  ├─ __init__.py
  └─ evaluation.py                  # evaluation metrics and helper functions

```

Notes:
- The `models/checkpoint/best_model.pt` is a saved PyTorch model (binary) used for quick inference or to resume training.
- The notebooks are the primary entry points for exploration and running experiments.

---

## Brief file/folder descriptions

- `Notebook_data_cleaning.ipynb` — Clean raw data, feature engineering, and produce the final dataset used for modeling. Contains steps for handling missing values, normalization, aggregation to customer-level features, and creation of the target label.

- `Notebook_prediction.ipynb` — Load the prepared dataset, train models (baseline models and the MLP in `models/`), tune hyperparameters, and evaluate performance. Contains visualization of results and comparison tables.

- `models/MLP.py` — Implementation of a simple feed-forward neural network (PyTorch) used as one of the models.

- `models/model.py` — Utilities for training, saving, loading models, and possibly wrappers that orchestrate training loops.

- `models/checkpoint/best_model.pt` — Best model checkpoint produced by prior training runs. Binary file, not human-readable.

- `utils/evaluation.py` — Functions to compute metrics (accuracy, precision, recall, F1, confusion matrix) and helper code to format results.

---

## Quick start

1. Install recommended packages (example):

```powershell
python -m pip install -r requirements.txt  # if you create one, or install manually
```

At minimum: pandas, numpy, scikit-learn, matplotlib, torch (PyTorch).

2. Open `Notebook_data_cleaning.ipynb` in Jupyter/VS Code to prepare the dataset.

3. Open `Notebook_prediction.ipynb` to run model training and evaluation. The notebook references `models/` and `utils/`.

---

## Verification

To verify the structure locally, run (from project root):

```powershell
# show tree on Windows PowerShell
Get-ChildItem -Recurse -Force | Format-List FullName
```

Or manually inspect the folders in your editor.

---

## Author

Project developed by Lucas.

---

If you want, I can also:
- Add a minimal `requirements.txt` based on imports found in the notebooks and scripts.
- Add a brief CONTRIBUTING or USAGE section showing exact notebook cells to run.
