
# ──────────────────────────────────────────────────────────────────────────────
# Functions adapted from:
# github.com/pkoerner6/Prediction-of-Aerobic-Biodegradability-of-Organic-Chemicals
# ──────────────────────────────────────────────────────────────────────────────
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import subprocess
import structlog
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from sklearn.model_selection import StratifiedKFold

log = structlog.get_logger()


def get_speciation_col_names() -> List[str]:
    return [
        "pka_acid_1", "pka_acid_2", "pka_acid_3", "pka_acid_4",
        "pka_base_1", "pka_base_2", "pka_base_3", "pka_base_4",
        "α_acid_0", "α_acid_1", "α_acid_2", "α_acid_3", "α_acid_4",
        "α_base_0", "α_base_1", "α_base_2", "α_base_3", "α_base_4",
    ]


def remove_smiles_with_incorrect_format(df: pd.DataFrame, col_name_smiles: str, prnt=False) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean[col_name_smiles] = df_clean[col_name_smiles].apply(lambda x: "nan" if "*" in x or "|" in x else x)
    df_clean = df_clean[df_clean[col_name_smiles] != "nan"]
    invalid_smiles = ["c1cccc1"]
    len_df_clean = len(df_clean)
    df_clean = df_clean[~df_clean[col_name_smiles].isin(invalid_smiles)]
    len_df_clean_after = len(df_clean)
    if len_df_clean_after < len_df_clean:
        log.warn("Removed this many SMILES with incorrect format", removed=len_df_clean - len_df_clean_after)
    df_clean.reset_index(inplace=True, drop=True)
    if prnt:
        log.warn("Removed this many data points because SMILES had incorrect format", removed=len(df) - len(df_clean))
    df_clean.reset_index(inplace=True, drop=True)
    return df_clean


def openbabel_convert(df: pd.DataFrame, input_type: str, column_name_input: str, output_type: str) -> pd.DataFrame:
    assert input_type in ("inchi", "smiles", "inchikey")
    assert output_type in ("inchi", "smiles", "inchikey")
    assert input_type != output_type

    input_data = list(df[column_name_input])
    with open("input.txt", "w") as f:
        for item in input_data:
            f.write(item + "\n")
    process = subprocess.run(
        ["obabel", f"-i{input_type}", "input.txt", f"-o{output_type}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    df[f"{output_type}_from_{column_name_input}"] = process.stdout.decode("utf-8").split("\n")[: len(df)]
    return df


def convert_to_maccs_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    mols = [AllChem.MolFromSmiles(smiles) for smiles in df["smiles"]]
    for index, value in enumerate(mols):
        if value is None:
            log.warn("This SMILES could not be converted to Mol file, deleting this datapoint", problematic_smiles=df.loc[index, "smiles"])
            df.drop(index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            del mols[index]
    df["fingerprint"] = [GetMACCSKeysFingerprint(mol) for mol in mols]
    return df


def bit_vec_to_lst_of_lst(df: pd.DataFrame, include_speciation: bool):
    def create_x_class(row) -> np.ndarray:
        speciation = []
        if include_speciation:
            speciation = row[get_speciation_col_names()].values.tolist()
        record_fp = np.array(row["fingerprint"]).tolist()
        return record_fp + speciation

    x_class = df.apply(create_x_class, axis=1)
    return x_class.to_list()


def create_input_classification(df_class: pd.DataFrame, include_speciation: bool, target_col: str) -> Tuple[np.ndarray, pd.Series]:
    df = convert_to_maccs_fingerprints(df_class)
    x_class = bit_vec_to_lst_of_lst(df, include_speciation)
    x_array = np.array(x_class, dtype=object)
    y = df[target_col]
    return x_array, y


def split_classification_df_with_fixed_test_set(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    nsplits: int,
    random_seed: int,
    cols: List[str],
    paper: bool,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    train_sets: List[pd.DataFrame] = []
    test_sets: List[pd.DataFrame] = []

    dfs = [df, df_test]
    for i, d in enumerate(dfs):
        if "inchi_from_smiles" not in d.columns:
            df_smiles_correct = remove_smiles_with_incorrect_format(df=d, col_name_smiles="smiles")
            dfs[i] = openbabel_convert(
                df=df_smiles_correct,
                input_type="smiles",
                column_name_input="smiles",
                output_type="inchi",
            )
    df = dfs[0][cols + ["inchi_from_smiles"]]
    df_test = dfs[1][cols + ["inchi_from_smiles"]]

    skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_seed)
    for _, test_index in skf.split(df_test[cols + ["inchi_from_smiles"]], df_test["y_true"]):
        df_test_set = df_test[df_test.index.isin(test_index)]
        train_set = df[~df["inchi_from_smiles"].isin(df_test_set["inchi_from_smiles"])]
        if paper:
            df_checked = pd.read_excel("datasets/chemical_speciation.xlsx", index_col=0)
            test_checked = df_checked[
                df_checked["env_smiles"].isin(df_test_set["smiles"]) |
                df_checked["inchi_from_smiles"].isin(df_test_set["inchi_from_smiles"])
            ]
            train_set = train_set[~(train_set["inchi_from_smiles"].isin(test_checked["inchi_from_smiles"]))]
            train_set = train_set[~(train_set["smiles"].isin(test_checked["env_smiles"]))]
            train_set = train_set.loc[~((train_set["cas"].isin(test_checked["cas"])) & (test_checked["cas"].notna())), :]
            train_set = train_set.loc[~((train_set["cas"].isin(df_test_set["cas"])) & (df_test_set["cas"].notna())), :]
        train_sets.append(train_set)
        test_sets.append(df_test_set)

    return train_sets, test_sets


def skf_class_fixed_testset(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    nsplits: int,
    random_seed: int,
    include_speciation: bool,
    cols: List[str],
    paper: bool,
    target_col: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[pd.DataFrame], List[int]]:
    train_sets, test_sets = split_classification_df_with_fixed_test_set(
        df=df,
        df_test=df_test,
        nsplits=nsplits,
        random_seed=random_seed,
        cols=cols,
        paper=paper,
    )
    x_train_fold_lst: List[np.ndarray] = []
    y_train_fold_lst: List[np.ndarray] = []
    x_test_fold_lst: List[np.ndarray] = []
    y_test_fold_lst: List[np.ndarray] = []
    df_test_lst: List[pd.DataFrame] = []
    test_set_sizes: List[int] = []

    for split in range(nsplits):
        train_fold = train_sets[split]
        test_fold = test_sets[split]
        x_train_fold_lst.append(train_fold["smiles"].values)
        y_train_fold_lst.append(train_fold[target_col].values)
        x_test_fold_lst.append(test_fold["smiles"].values)
        y_test_fold_lst.append(test_fold[target_col].values)
        df_test_lst.append(test_fold.copy())
        test_set_sizes.append(len(test_fold))

    return x_train_fold_lst, y_train_fold_lst, x_test_fold_lst, y_test_fold_lst, df_test_lst, test_set_sizes


# ──────────────────────────────────────────────────────────────────────────────
# Apply splitting: df_curated_final (train) vs df_curated_scs (fixed test set)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    import download_data

    df_curated_biowin, df_curated_final, df_curated_scs = download_data.load_curated_datasets()

    print(f"df_curated_final (train pool): {df_curated_final.shape}")
    print(f"df_curated_scs   (test set):   {df_curated_scs.shape}")

    cols = ["cas", "smiles", "y_true"]

    print("\nRunning 5-fold cross-validation with fixed test set (df_curated_scs)...")
    x_train_folds, y_train_folds, x_test_folds, y_test_folds, df_test_folds, test_sizes = skf_class_fixed_testset(
        df=df_curated_final,
        df_test=df_curated_scs,
        nsplits=5,
        random_seed=42,
        include_speciation=False,
        cols=cols,
        paper=False,
        target_col="y_true",
    )

    print("\nResults per fold:")
    for i in range(5):
        print(f"  Fold {i+1}: x_train={x_train_folds[i].shape}, x_test={x_test_folds[i].shape}, test_size={test_sizes[i]}")
