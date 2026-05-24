"""OTU/feature abundance table container.

R reference: phyloseq::otu_table(object)
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


class OtuTable:
    """Stores an OTU/feature abundance table with orientation tracking.

    Internally stores dense data as a ``pd.DataFrame`` and sparse data (density
    < 50 %) as a ``scipy.sparse.csr_matrix`` with separate index/column arrays.

    R reference: phyloseq::otu_table(object, taxa_are_rows)
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame, sp.spmatrix, list],  # noqa: UP007
        taxa_are_rows: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        data:
            Abundance matrix. Accepted types: ``pd.DataFrame``, ``np.ndarray``,
            any ``scipy.sparse`` matrix, or a list-of-lists.
        taxa_are_rows:
            If ``True`` (default), rows represent taxa and columns represent
            samples.
        """
        if isinstance(data, pd.DataFrame):
            row_idx: pd.Index = data.index
            col_idx: pd.Index = data.columns
            raw: np.ndarray | sp.spmatrix = data.values
            is_sparse_input = False
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(f"Expected 2-D array, got shape {data.shape}")
            raw = data
            row_idx = pd.RangeIndex(data.shape[0])
            col_idx = pd.RangeIndex(data.shape[1])
            is_sparse_input = False
        elif sp.issparse(data):
            raw = data
            row_idx = pd.RangeIndex(data.shape[0])
            col_idx = pd.RangeIndex(data.shape[1])
            is_sparse_input = True
        elif isinstance(data, list):
            arr = np.array(data, dtype=float)
            if arr.ndim != 2:
                raise ValueError("list-of-lists input must be 2-D")
            raw = arr
            row_idx = pd.RangeIndex(arr.shape[0])
            col_idx = pd.RangeIndex(arr.shape[1])
            is_sparse_input = False
        else:
            raise TypeError(f"Unsupported OtuTable data type: {type(data)!r}")

        # Compute density to decide storage format
        if sp.issparse(raw):
            nelem = raw.shape[0] * raw.shape[1]
            nnz = raw.nnz  # type: ignore[union-attr]
        else:
            arr2d = np.asarray(raw)
            nelem = arr2d.size
            nnz = int(np.count_nonzero(arr2d))

        density = nnz / nelem if nelem > 0 else 1.0

        self._row_index: pd.Index = pd.Index(row_idx)
        self._col_index: pd.Index = pd.Index(col_idx)
        self._taxa_are_rows: bool = taxa_are_rows
        self._df: pd.DataFrame | None = None
        self._sparse: sp.csr_matrix | None = None

        if density < 0.5 or is_sparse_input:
            csr: sp.csr_matrix = (
                raw.tocsr() if sp.issparse(raw) else sp.csr_matrix(np.asarray(raw))  # type: ignore[arg-type]
            )
            self._sparse = csr
        else:
            dense = np.asarray(raw.toarray() if sp.issparse(raw) else raw)  # type: ignore[union-attr]
            self._df = pd.DataFrame(dense, index=self._row_index, columns=self._col_index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _is_sparse(self) -> bool:
        return self._sparse is not None

    def _to_numpy(self) -> np.ndarray:
        if self._df is not None:
            return self._df.values  # type: ignore[return-value]
        assert self._sparse is not None
        return self._sparse.toarray()

    def to_dataframe(self) -> pd.DataFrame:
        """Return the abundance matrix as a ``pd.DataFrame`` in current orientation.

        R reference: as(otu_table(x), "matrix") then as.data.frame()
        """
        if self._df is not None:
            return self._df
        assert self._sparse is not None
        return pd.DataFrame(
            self._sparse.toarray(),
            index=self._row_index,
            columns=self._col_index,
        )

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------

    @property
    def taxa_are_rows(self) -> bool:
        """Whether taxa occupy rows (``True``) or columns (``False``)."""
        return self._taxa_are_rows

    @taxa_are_rows.setter
    def taxa_are_rows(self, value: bool) -> None:
        if value == self._taxa_are_rows:
            return
        if self._df is not None:
            self._df = self._df.T
        else:
            assert self._sparse is not None
            self._sparse = self._sparse.T.tocsr()
        self._row_index, self._col_index = self._col_index, self._row_index
        self._taxa_are_rows = value

    # ------------------------------------------------------------------
    # Names and dimensions
    # ------------------------------------------------------------------

    @property
    def taxa_names(self) -> pd.Index:
        """Taxa (OTU/ASV) identifiers.

        R reference: taxa_names(x)
        """
        return self._row_index if self._taxa_are_rows else self._col_index

    @taxa_names.setter
    def taxa_names(self, names: pd.Index | list) -> None:
        new_idx = pd.Index(names)
        if self._taxa_are_rows:
            self._row_index = new_idx
            if self._df is not None:
                self._df.index = new_idx
        else:
            self._col_index = new_idx
            if self._df is not None:
                self._df.columns = new_idx

    @property
    def sample_names(self) -> pd.Index:
        """Sample identifiers.

        R reference: sample_names(x)
        """
        return self._col_index if self._taxa_are_rows else self._row_index

    @sample_names.setter
    def sample_names(self, names: pd.Index | list) -> None:
        new_idx = pd.Index(names)
        if self._taxa_are_rows:
            self._col_index = new_idx
            if self._df is not None:
                self._df.columns = new_idx
        else:
            self._row_index = new_idx
            if self._df is not None:
                self._df.index = new_idx

    @property
    def ntaxa(self) -> int:
        """Number of taxa.

        R reference: ntaxa(x)
        """
        return len(self._row_index) if self._taxa_are_rows else len(self._col_index)

    @property
    def nsamples(self) -> int:
        """Number of samples.

        R reference: nsamples(x)
        """
        return len(self._col_index) if self._taxa_are_rows else len(self._row_index)

    # ------------------------------------------------------------------
    # Sums
    # ------------------------------------------------------------------

    def taxa_sums(self) -> pd.Series:
        """Sum of abundances across all samples for each taxon.

        R reference: taxa_sums(x)
        """
        arr = self._to_numpy()
        sums: np.ndarray = arr.sum(axis=1) if self._taxa_are_rows else arr.sum(axis=0)
        return pd.Series(sums, index=self.taxa_names, name="taxa_sums")

    def sample_sums(self) -> pd.Series:
        """Sum of abundances across all taxa for each sample.

        R reference: sample_sums(x)
        """
        arr = self._to_numpy()
        sums = arr.sum(axis=0) if self._taxa_are_rows else arr.sum(axis=1)
        return pd.Series(sums, index=self.sample_names, name="sample_sums")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        storage = "sparse" if self._is_sparse else "dense"
        return (
            f"OtuTable({self.ntaxa} taxa × {self.nsamples} samples, "
            f"taxa_are_rows={self._taxa_are_rows}, {storage})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OtuTable):
            return NotImplemented
        if self._taxa_are_rows != other._taxa_are_rows:
            return False
        if not self._row_index.equals(other._row_index):
            return False
        if not self._col_index.equals(other._col_index):
            return False
        return bool(np.allclose(self._to_numpy(), other._to_numpy()))

    def copy(self) -> OtuTable:
        """Return a deep copy.

        R reference: otu_table(x) <- otu_table(x) (effectively)
        """
        new = OtuTable.__new__(OtuTable)
        new._taxa_are_rows = self._taxa_are_rows
        new._row_index = self._row_index.copy()
        new._col_index = self._col_index.copy()
        new._df = self._df.copy() if self._df is not None else None
        new._sparse = self._sparse.copy() if self._sparse is not None else None
        return new
