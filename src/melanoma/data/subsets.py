"""Class-stratified subsetting that keeps every lesion intact."""

import pandas as pd


def stratified_subset(frame: pd.DataFrame, size, seed: int = 42) -> pd.DataFrame:
    """Sample an approximately class-stratified subset without splitting lesions.

    The target size is approximate: whole lesions are selected, so a lesion
    with several images contributes all of them. Every class present in the
    input is represented in the output.

    Raises ValueError for an empty frame, a non-positive size, or a size
    smaller than the number of classes.
    """
    if frame.empty:
        raise ValueError("Cannot subset an empty metadata frame")
    for column in ("dx", "lesion_id"):
        if column not in frame.columns:
            raise ValueError(f"Frame must contain a '{column}' column")
    if size is None or size >= len(frame):
        return frame.copy()
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"size must be a positive integer or None, got {size!r}")
    n_classes = frame["dx"].nunique()
    if size < n_classes:
        raise ValueError(f"size must be at least the number of classes ({n_classes}), got {size}")

    fraction = size / len(frame)
    pieces = []
    for _, group in frame.groupby("dx"):
        lesion_ids = group["lesion_id"].drop_duplicates()
        n_lesions = max(1, int(round(len(lesion_ids) * fraction)))
        n_lesions = min(n_lesions, len(lesion_ids))
        chosen = lesion_ids.sample(n=n_lesions, random_state=seed)
        pieces.append(group[group["lesion_id"].isin(chosen)])
    subset = pd.concat(pieces)
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
