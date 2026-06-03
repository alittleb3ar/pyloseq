"""Per-sample metadata container."""

from __future__ import annotations

from typing import cast

import pandas as pd

from pyloseq._validation import require_unique


class SampleData:
    """Wraps a ``pd.DataFrame`` of per-sample metadata.

    The DataFrame index must be sample identifiers, and must be unique.

    R reference: phyloseq::sample_data(object)
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"SampleData requires a pd.DataFrame, got {type(data)!r}")
        require_unique(data.index, "Sample names")
        self._df: pd.DataFrame = data.copy()
        self._df.index.name = None

    @property
    def sample_names(self) -> pd.Index:
        """Sample identifiers (DataFrame index).

        R reference: sample_names(x)
        """
        return self._df.index

    # Backwards-compatible alias. ``names`` was the original accessor; prefer
    # ``sample_names`` so the component is self-describing outside the container.
    @property
    def names(self) -> pd.Index:
        """Deprecated alias for :attr:`sample_names`."""
        return self.sample_names

    @property
    def variables(self) -> pd.Index:
        """Sample variable names (DataFrame columns).

        R reference: sample_variables(x)
        """
        return self._df.columns

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        R reference: as(sample_data(x), "data.frame")
        """
        return cast(pd.DataFrame, self._df.copy())

    def copy(self) -> SampleData:
        """Return a deep copy of this SampleData."""
        return SampleData(self._df.copy())

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f"SampleData({len(self._df)} samples × {len(self._df.columns)} variables)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SampleData):
            return NotImplemented
        return bool(self._df.equals(other._df))
