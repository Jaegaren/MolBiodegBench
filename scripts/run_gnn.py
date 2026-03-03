import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from src.biodeg.graph import load_graphs
from src.biodeg.model import GCN

DATA_PATH = "data/class_curated_final.csv"
EPOCHS = 50
LR = 1e-3
BATCH = 32
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = load_graphs(DATA_PATH)
labels = [g.y.item() for g in graphs]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(graphs, labels)):
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=BATCH, shuffle=True)
    test_loader = DataLoader([graphs[i] for i in test_idx],  batch_size=BATCH)

    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        model.train()


        for batch in train_loader:
            batch = batch.to(device)
            loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            predictions += out.argmax(dim=1).cpu().tolist()
            targets += batch.y.cpu().tolist()

    ba = balanced_accuracy_score(targets, predictions)
    results.append(ba)
    print(f"Fold {fold+1}: balanced accuracy = {ba}")

print(f"Mean BA: {np.mean(results)} ± {np.std(results)}")