import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from src.biodeg.graph import smiles_to_graphs
from src.biodeg.model import GCN
from download_data import load_curated_datasets
from preprocess import skf_class_fixed_testset

from sklearn.metrics import confusion_matrix
import seaborn as sns

EPOCHS = 150
LR = 1e-3
BATCH = 32
SEED = 42
WEIGHT_DECAY = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_biowin, df_final, df_scs = load_curated_datasets()

x_train_folds, y_train_folds, x_test_folds, y_test_folds, _, _ = skf_class_fixed_testset(
    df=df_final,
    df_test=df_biowin,
    nsplits=5,
    random_seed=SEED,
    include_speciation=False,
    cols=["cas", "smiles", "y_true"],
    paper=False,
    target_col="y_true",
)

results = []
all_train_losses = []
all_test_losses  = []

all_preds = []
all_targets = []

print("Building graphs...")
train_graphs_all = [smiles_to_graphs(x_train_folds[i], y_train_folds[i]) for i in range(5)]
test_graphs_all = [smiles_to_graphs(x_test_folds[i],  y_test_folds[i]) for i in range(5)]
print("Done!")

for fold in range(5):
    train_graphs = train_graphs_all[fold]
    test_graphs  = test_graphs_all[fold]

    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH)

    model = GCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    test_losses  = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch.x, batch.edge_index, batch.batch), batch.y).item()
        test_losses.append(val_loss / len(test_loader))

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

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

    all_preds += predictions
    all_targets += targets


print(f"Mean BA: {np.mean(results)} ± {np.std(results)}")

# --- Loss curves ---
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for fold, ax in enumerate(axes):
    ax.plot(epochs, all_train_losses[fold], label="Train loss")
    ax.plot(epochs, all_test_losses[fold],  label="Test loss")
    ax.set_title(f"Fold {fold + 1}")
    ax.set_xlabel("Epoch")
    if fold == 0:
        ax.set_ylabel("Loss")
    ax.legend()
fig.suptitle("Train vs Test loss per fold")
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()


print(f"all_preds length: {len(all_preds)}")
print(f"all_targets length: {len(all_targets)}")

# --- Confusion matrix ---
cm = confusion_matrix(all_targets, all_preds)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not biodegradable', 'Biodegradable'],
            yticklabels=['Not biodegradable', 'Biodegradable'])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix — aggregated over 5 folds')
plt.tight_layout()
#plt.savefig("confusion_matrix.png", dpi=150)
plt.savefig("confusion_matrix_biowin.png", dpi=150)
plt.show()
