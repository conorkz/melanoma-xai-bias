"""Patient or lesion level train/validation/test splitting without leakage."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_splits(
    frame: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    group_col: str = "lesion_id",
) -> pd.DataFrame:
    """Split a metadata frame into train/val/test keeping groups intact.

    The same lesion (group) never appears in more than one split, which
    prevents leakage when a lesion has multiple images. Returns a copy of the
    frame with an added 'split' column containing 'train', 'val', or 'test'.

    Raises ValueError for invalid fractions, a missing or empty group column,
    NaN groups, or too few unique groups for the requested split.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1, got {val_fraction + test_fraction}"
        )
    if frame.empty:
        raise ValueError("Cannot split an empty metadata frame")
    if group_col not in frame.columns:
        raise ValueError(f"Group column '{group_col}' not found in metadata")
    if frame[group_col].isna().any():
        raise ValueError(f"Group column '{group_col}' contains NaN values")
    n_groups = frame[group_col].nunique()
    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups to split, got {n_groups}")

    frame = frame.copy().reset_index(drop=True)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    rest_idx, test_idx = next(splitter_test.split(frame, groups=frame[group_col]))

    rest = frame.iloc[rest_idx]
    val_share_of_rest = val_fraction / (1.0 - test_fraction)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_share_of_rest, random_state=seed)
    train_idx, val_idx = next(splitter_val.split(rest, groups=rest[group_col]))

    frame["split"] = "train"
    frame.loc[rest.index[val_idx], "split"] = "val"
    frame.loc[test_idx, "split"] = "test"
    return frame
