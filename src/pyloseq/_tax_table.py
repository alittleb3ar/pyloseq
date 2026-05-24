"""Taxonomic classification table container.

R reference: phyloseq::tax_table(object)
"""

from __future__ import annotations

from typing import cast

import pandas as pd


class TaxTable:
    """Wraps a ``pd.DataFrame`` of taxonomic classifications.

    The DataFrame index must be taxa identifiers; columns are rank names
    (e.g. ``["Kingdom", "Phylum", "Class", "Order", "Family", "Genus",
    "Species"]``).  Rank names are user-supplied and not hardcoded.

    R reference: phyloseq::tax_table(object)
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"TaxTable requires a pd.DataFrame, got {type(data)!r}")
        self._df = data.copy()
        self._df.index.name = None

    @property
    def names(self) -> pd.Index:
        """Taxon identifiers (DataFrame index).

        R reference: taxa_names(x)
        """
        return cast(pd.Index, self._df.index)

    @property
    def rank_names(self) -> list[str]:
        """Taxonomic rank names (column names).

        R reference: rank_names(x)
        """
        return list(self._df.columns)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        R reference: as(tax_table(x), "matrix") then as.data.frame()
        """
        return cast(pd.DataFrame, self._df.copy())

    def copy(self) -> TaxTable:
        """Return a deep copy of this TaxTable."""
        return TaxTable(self._df.copy())

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        ranks = ", ".join(self.rank_names[:3])
        if len(self.rank_names) > 3:
            ranks += ", …"
        return f"TaxTable({len(self._df)} taxa × {len(self._df.columns)} ranks [{ranks}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaxTable):
            return NotImplemented
        return bool(self._df.equals(other._df))
