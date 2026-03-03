from pathlib import Path
import pandas as pd


def find_repo_root(start: Path = None) -> Path:
    start = Path(__file__).resolve() if start is None else Path(start).resolve()
    for p in [start] + list(start.parents):
        if (p / ".git").exists() or (p / ".gitignore").exists():
            return p
    raise FileNotFoundError("Git repository root not found from path: " + str(start))


def load_curated_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repo_root = find_repo_root()
    data_dir = repo_root / "data"
    df_curated_biowin = pd.read_csv(data_dir / "class_curated_biowin.csv", index_col=0)
    df_curated_final = pd.read_csv(data_dir / "class_curated_final.csv", index_col=0)
    df_curated_scs = pd.read_csv(data_dir / "class_curated_scs.csv", index_col=0)
    return df_curated_biowin, df_curated_final, df_curated_scs


if __name__ == "__main__":
    df_curated_biowin, df_curated_final, df_curated_scs = load_curated_datasets()
    print(f"df_curated_biowin: {df_curated_biowin.shape} | columns: {list(df_curated_biowin.columns)}")
    print(f"df_curated_final:  {df_curated_final.shape} | columns: {list(df_curated_final.columns)}")
    print(f"df_curated_scs:    {df_curated_scs.shape} | columns: {list(df_curated_scs.columns)}")
