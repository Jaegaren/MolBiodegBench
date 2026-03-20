# MolBiodegBench

Predicting biodegradability of organic chemicals using graph neural networks,
comparing GNN-based approaches against classical fingerprint-based baselines.

## Project Structure
```
MolBiodegBench/
├── src/biodeg/
│   ├── model.py              # GCN model definition
│   ├── graph.py              # Molecular graph featurisation
│   └── config.py             # Configuration handling
├── scripts/
│   ├── run_gnn.py            # GCN training and evaluation
│   ├── run_fingerprint_baseline.py  # XGBoost/fingerprint baseline
│   ├── preprocess.py         # Data preprocessing
│   ├── evaluate.py           # Evaluation utilities
│   └── download_data.py      # Data download script
├── legacy_models/
│   ├── models.ipynb          # KNN, SVM, PLS-DA (Mansouri et al. recreation)
│   └── environment_mansouri.yml
├── finetuning/
│   ├── new_GNN.ipynb         # CheMeleon finetuning notebook
│   └── environment_gnn.yml
├── configs/
│   ├── baseline.yaml         # Fingerprint baseline config
│   └── gnn.yaml              # GCN config
├── data/                     # Place datasets here (see Data section)
├── results/                  # Output figures and metrics
└── tests/
    └── test_smoke.py
```

## Setup

### Custom GCN and baselines
```bash
conda env create -f environment.yml
conda activate GNN_chem2
pip install -e .
```

### Legacy ML models (Mansouri recreation)
```bash
conda env create -f legacy_models/environment_mansouri.yml
conda activate mansouri_models
```

### CheMeleon finetuning
```bash
conda env create -f finetuning/environment_gnn.yml
conda activate GNN_chem2
```

## Data

Datasets are from Körner et al. (2024) and are not included in this repository.
Download them and place in the `data/` folder. Expected files:
- `class_curated_biowin.csv`
- `class_curated_scs.csv`
- `class_curated_final.csv`

For the legacy models, `dataset.xlsx` from Mansouri et al. is also required.

## Running the code

**Custom GCN:**
```bash
python scripts/run_gnn.py
```

**Fingerprint baseline:**
```bash
python scripts/run_fingerprint_baseline.py
```

**Legacy ML models:** Open and run `legacy_models/models.ipynb`

**CheMeleon finetuning:** Open and run `finetuning/new_GNN.ipynb`

## Expected outputs

Results are saved to `results/` and `scripts/`:
- Confusion matrices per model and test set
- Loss curves per fold for the GCN
- Balanced accuracy scores printed to console
