from rdkit import Chem
import torch
from torch_geometric.data import Data
import pandas as pd

def mol_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atoms = []

    for atom in mol.GetAtoms():
        atoms.append([atom.GetAtomicNum(), atom.GetDegree(), int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(atom.IsInRing()),])

    edges = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    x = torch.tensor(atoms, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def load_graphs(csv_path):
    df = pd.read_csv(csv_path)

    graphs = []

    for _, row in df.iterrows():
        g = mol_to_graph(row["smiles"], int(row["label"]))
        if g is not None:
            graphs.append(g)

    print(f"Loaded {len(graphs)} molecules from {csv_path}")
    return graphs



