"""Load and prepare the PriorityLab email dataset (no model fitting)."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "weijianzhg/email-triage-action-seed"
REQUIRED_COLUMNS = {"subject", "body", "priority", "category", "action", "source_round"}
PRIORITY_TO_IMPORTANT = {
    "urgent": 1,
    "high": 1,
    "medium": 0,
    "low": 0,
}
RANDOM_STATE = 42


def load_raw_dataframe(split: str = "train") -> pd.DataFrame:
    raw = load_dataset(DATASET_ID, split=split)
    df = raw.to_pandas()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary labels and build the text feature column."""
    out = df.copy()
    out["priority_norm"] = out["priority"].astype("string").str.strip().str.lower()
    unknown = set(out["priority_norm"].dropna().unique()) - set(PRIORITY_TO_IMPORTANT)
    if unknown or out["priority_norm"].isna().any():
        raise ValueError(f"Bad or missing priority labels: {unknown}")

    out["important"] = out["priority_norm"].map(PRIORITY_TO_IMPORTANT).astype(int)
    if out["important"].nunique() != 2:
        raise ValueError("Both binary classes must be present")

    out["subject_clean"] = out["subject"].fillna("").astype(str).str.strip()
    out["body_clean"] = out["body"].fillna("").astype(str).str.strip()
    out["text"] = "subject: " + out["subject_clean"] + " body: " + out["body_clean"]

    # Exact-text dedupe keeps the first row; document conflict count before dropping.
    conflict_texts = int((out.groupby("text")["important"].nunique() > 1).sum())
    if conflict_texts:
        # Keep first occurrence; contradictory pairs should be reviewed in EDA.
        pass
    out = out.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return out


def stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 stratified split on `important` (approximate; val carved from train)."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["important"],
        random_state=random_state,
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val["important"],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_prepared_splits(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = prepare_modeling_frame(load_raw_dataframe())
    return stratified_splits(df, test_size=test_size, val_size=val_size, random_state=random_state)
