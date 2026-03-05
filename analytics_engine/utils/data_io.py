from pathlib import Path

import pandas as pd


def load_dataset_frame(file_path: str, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def persist_dataframe(df: pd.DataFrame, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.suffix.lower() == ".csv":
        df.to_csv(target_path, index=False)
    else:
        df.to_excel(target_path, index=False)
