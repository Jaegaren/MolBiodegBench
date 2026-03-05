from rdkit import Chem, RDLogger
import torch
import pandas as pd
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')


def standardize_smiles(smiles):

    try:

        #step 1
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        else:
            None

        if mol is None:
            return None

        #step 2
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)

        #step 3
        mol = rdMolStandardize.Uncharger().uncharge(mol)

        #step 4
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        return smiles


def mol_to_graph(smiles, label):
    smiles = standardize_smiles(smiles)
    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atoms = []

    for atom in mol.GetAtoms():
        atoms.append([atom.GetAtomicNum(), atom.GetDegree(), int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(atom.IsInRing())])

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


def smiles_to_graphs(smiles_array, labels_array):
    graphs = []
    for smiles, label in zip(smiles_array, labels_array):
        g = mol_to_graph(smiles, int(label))
        if g is not None:
            graphs.append(g)
    return graphs



