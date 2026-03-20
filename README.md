# MolBiodegBench

Predicting biodegradability of organic chemicals using graph neural networks.
This repository contains code for three modelling approaches: recreated legacy 
ML models, a custom GCN trained from scratch, and a finetuned CheMeleon GNN.

## Project Structure
```
MolBiodegBench/
├── legacy_models/
│   ├── models.ipynb              # KNN, SVM, PLS-DA recreation of Mansouri et al.
│   └── environment_mansouri.yml  # Conda environment for legacy models
├── finetuning/
│   ├── new_GNN.ipynb             # CheMeleon finetuning notebook
│   └── environment_gnn.yml       # Conda environment for finetuning
├── scripts/
│   ├── run_gnn.py                # GCN training and evaluation script
│   ├── loss_curves.png           # Training/test loss per fold
│   ├── confusion_matrix.png      # Confusion matrix - SCS dataset
│   └── confusion_matrix_biowin.png # Confusion matrix - Biowin dataset
└── data/
    └── README.md                 # Instructions for obtaining datasets
```

## Setup

### Legacy ML models (Mansouri recreation)
```bash
conda env create -f legacy_models/environment_mansouri.yml
conda activate mansouri_models
```

### Custom GCN
```bash
conda env create -f environment.yml
conda activate GNN_chem2
```

### CheMeleon finetuning
```bash
conda env create -f finetuning/environment_gnn.yml
conda activate GNN_chem2
```

## Data

The datasets are from Körner et al. (2024) and are not included in this 
repository. Download them from their paper or repository and place them in 
the `data/` folder. The expected files are:
- `class_curated_biowin.csv`
- `class_curated_scs.csv`
- `class_curated_final.csv`

For the legacy models, `dataset.xlsx` from Mansouri et al. is also required.

## Running the code

**Legacy ML models:** Open and run `legacy_models/models.ipynb`

**Custom GCN:** 
```bash
python scripts/run_gnn.py
```

**CheMeleon finetuning:** Open and run `finetuning/new_GNN.ipynb`

## Expected outputs

- Balanced accuracy scores for each model on both test sets
- Confusion matrices saved to `scripts/`
- Loss curves saved to `scripts/`
