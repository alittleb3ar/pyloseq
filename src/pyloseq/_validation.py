"""Small shared validation helpers used across component containers."""

from __future__ import annotations

import pandas as pd


def require_unique(idx: pd.Index, label: str) -> None:
    """Raise ``ValueError`` if ``idx`` contains duplicate labels.

    Parameters
    ----------
    idx:
        Index to check.
    label:
        Human-readable name for the thing being checked (e.g. ``"Sample names"``),
        used in the error message.
    """
    if not idx.is_unique:
        dups = idx[idx.duplicated()].unique().tolist()
        raise ValueError(f"{label} must be unique; duplicates: {dups}")
