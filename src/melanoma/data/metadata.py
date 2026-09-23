"""HAM10000 metadata loading and label mapping."""

from pathlib import Path

import pandas as pd

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT_CLASS = "mel"

REQUIRED_COLUMNS = ["image_id", "dx", "lesion_id"]


def load_metadata(path) -> pd.DataFrame:
    """Load and validate a HAM10000 metadata CSV.

    Raises FileNotFoundError if the CSV is missing, ValueError if required
    columns are absent, labels are unknown, or image ids are duplicated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Metadata CSV contains no rows")
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].unique()
        raise ValueError(f"Duplicate image_id values in metadata: {list(duplicates[:5])}")
    frame = frame.copy()
    frame["dx"] = frame["dx"].astype(str).str.strip().str.lower()
    unknown = sorted(set(frame["dx"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown diagnosis labels found: {unknown}")
    return frame


def class_counts(frame: pd.DataFrame) -> pd.Series:
    """Return diagnosis counts sorted in ascending order."""
    return frame["dx"].value_counts().sort_values()


def to_binary_labels(frame: pd.DataFrame) -> pd.Series:
    """Map diagnoses to binary labels: melanoma is 1, everything else is 0."""
    return (frame["dx"] == MALIGNANT_CLASS).astype(int)


def to_class_indices(frame: pd.DataFrame) -> pd.Series:
    """Map diagnoses to contiguous class indices following CLASS_NAMES order."""
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    return frame["dx"].map(mapping).astype(int)
