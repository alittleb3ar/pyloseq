"""Ordination dispatcher mirroring R phyloseq::ordinate.

R reference: phyloseq::ordinate(physeq, method, distance, formula, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyloseq._exceptions import pyloseqValidationError
from pyloseq._manipulation import _otu_samples_rows

if TYPE_CHECKING:
    from skbio.stats.ordination import OrdinationResults

    from pyloseq._phyloseq import Phyloseq

_SUPPORTED_METHODS = {
    "PCoA",
    "MDS",  # aliases — PCoA via scikit-bio pcoa()
    "NMDS",  # Non-metric MDS via scikit-bio nmds()
    "CCA",  # Canonical Correspondence Analysis
    "RDA",  # Redundancy Analysis
    "CAP",  # Constrained Analysis of Principal Coordinates
    "DPCoA",  # Double Principal Coordinates Analysis
    "DCA",  # Detrended Correspondence Analysis (not implemented)
}
# Pre-computed for O(1) lookup after upper-casing the caller's method string
_SUPPORTED_METHODS_UPPER = {s.upper() for s in _SUPPORTED_METHODS}


def ordinate(
    ps: Phyloseq,
    method: str = "PCoA",
    distance: str | Any = "bray",
    formula: str | None = None,
    **kwargs: Any,
) -> OrdinationResults:
    """Perform multivariate ordination on a Phyloseq object.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    method:
        Ordination method: ``"PCoA"``/``"MDS"``, ``"NMDS"``, ``"CCA"``,
        ``"RDA"``, ``"CAP"``, ``"DPCoA"``, or ``"DCA"``.
    distance:
        Distance method string (passed to :func:`pyloseq.distance`) or a
        pre-computed ``skbio.DistanceMatrix``.  Ignored for ``CCA`` and
        ``RDA``.
    formula:
        Model formula string referencing columns of ``sample_data`` (e.g.
        ``"~SampleType + Depth"``).  Required for ``CCA``, ``RDA``, ``CAP``.
    **kwargs:
        Passed to the underlying ordination function.

    Returns
    -------
    skbio.stats.ordination.OrdinationResults

    R reference: ordinate(physeq, method, distance, formula, ...)
    """
    m = method.upper()

    if m not in _SUPPORTED_METHODS_UPPER:
        raise pyloseqValidationError(
            f"Unknown ordination method: '{method}'. Supported: {sorted(_SUPPORTED_METHODS)}"
        )

    if m in ("PCOA", "MDS"):
        return _pcoa(ps, distance, **kwargs)
    if m == "NMDS":
        return _nmds(ps, distance, **kwargs)
    if m == "CCA":
        return _cca(ps, formula, **kwargs)
    if m == "RDA":
        return _rda(ps, formula, **kwargs)
    if m == "CAP":
        return _cap(ps, distance, formula, **kwargs)
    if m == "DPCOA":
        return _dpcoa_ordinate(ps, **kwargs)
    if m == "DCA":
        raise NotImplementedError(
            "DCA (Detrended Correspondence Analysis) is not implemented. "
            "Use CCA, RDA, or PCoA instead."
        )

    raise pyloseqValidationError(f"Unhandled method: '{method}'")  # unreachable


# ---------------------------------------------------------------------------
# PCoA / MDS
# ---------------------------------------------------------------------------


def _resolve_dm(ps: Phyloseq, distance: str | Any) -> Any:
    """Compute or return a distance matrix."""
    from skbio.stats.distance import DistanceMatrix

    if isinstance(distance, DistanceMatrix):
        return distance
    if isinstance(distance, str):
        from pyloseq._distances import distance as _distance  # noqa: PLC0415

        return _distance(ps, distance)
    raise TypeError(f"distance must be a string or DistanceMatrix, got {type(distance)!r}")


def _pcoa(ps: Phyloseq, distance: str | Any, **kwargs: Any) -> Any:
    """Principal Coordinates Analysis (PCoA / MDS).

    R reference: ordinate(physeq, "PCoA", distance)
    """
    from skbio.stats.ordination import pcoa

    dm = _resolve_dm(ps, distance)
    result = pcoa(dm, **kwargs)
    return result


# ---------------------------------------------------------------------------
# NMDS
# ---------------------------------------------------------------------------


def _nmds(ps: Phyloseq, distance: str | Any, **kwargs: Any) -> Any:
    """Non-metric multidimensional scaling.

    Uses scikit-bio's ``nmds`` if available; falls back to scipy smacof.

    R reference: ordinate(physeq, "NMDS", distance)
    """
    dm = _resolve_dm(ps, distance)

    # Try scikit-bio nmds first
    try:
        from skbio.stats.ordination import nmds

        return nmds(dm, **kwargs)
    except (ImportError, AttributeError):
        pass

    # Fall back: use sklearn MDS with precomputed square distance matrix
    from skbio.stats.ordination import OrdinationResults

    n_dims = kwargs.get("number_of_dimensions", 2)
    dist_sq = np.array(dm.data)  # already square — do NOT squareform

    try:
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=n_dims,
            dissimilarity="precomputed",
            random_state=kwargs.get("random_state", 42),
            metric=False,
            n_init=1,
        )
        coords = mds.fit_transform(dist_sq)
    except ImportError:
        # Double centering for metric MDS (classical MDS approximation)
        H = np.eye(len(dist_sq)) - np.ones_like(dist_sq) / len(dist_sq)
        B = -0.5 * H @ (dist_sq**2) @ H
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(-vals)
        vals, vecs = vals[idx], vecs[:, idx]
        vals_pos = np.maximum(vals[:n_dims], 0)
        coords = vecs[:, :n_dims] * np.sqrt(vals_pos)

    sample_ids = list(dm.ids)
    col_names = [f"NMDS{i + 1}" for i in range(n_dims)]
    samples_df = pd.DataFrame(coords, index=sample_ids, columns=col_names)

    return OrdinationResults(
        short_method_name="NMDS",
        long_method_name="Nonmetric Multidimensional Scaling",
        eigvals=pd.Series([0.0] * n_dims, index=col_names),
        samples=samples_df,
        proportion_explained=pd.Series([np.nan] * n_dims, index=col_names),
    )


# ---------------------------------------------------------------------------
# CCA / RDA
# ---------------------------------------------------------------------------


def _parse_formula(ps: Phyloseq, formula: str | None) -> pd.DataFrame:
    """Extract sample-data columns from a formula string.

    Parses ``"~Var1 + Var2 + ..."`` into a numeric DataFrame, dummy-encoding
    any non-numeric columns.
    """
    if ps.sample_data is None:
        raise pyloseqValidationError("CCA/RDA requires sample_data")
    if formula is None:
        raise pyloseqValidationError("CCA/RDA requires a formula string")

    sam_df = ps.sample_data.to_frame()
    terms = [t.strip() for t in formula.lstrip("~").split("+")]
    missing = [t for t in terms if t not in sam_df.columns]
    if missing:
        raise pyloseqValidationError(f"Formula terms not found in sample_data: {missing}")
    sub = sam_df[terms]
    # Dummy-encode categorical/object columns so RDA/CCA get numeric input
    return pd.get_dummies(sub, drop_first=True).astype(float)


def _cca(ps: Phyloseq, formula: str | None, **kwargs: Any) -> Any:
    """Canonical Correspondence Analysis.

    R reference: ordinate(physeq, "CCA", formula=~Var)
    """
    from skbio.stats.ordination import cca

    x_df = _parse_formula(ps, formula)

    otu_df = _otu_samples_rows(ps)

    # Align to shared samples
    shared = otu_df.index.intersection(x_df.index)
    return cca(otu_df.loc[shared], x_df.loc[shared], **kwargs)


def _rda(ps: Phyloseq, formula: str | None, **kwargs: Any) -> Any:
    """Redundancy Analysis.

    R reference: ordinate(physeq, "RDA", formula=~Var)
    """
    from skbio.stats.ordination import rda

    x_df = _parse_formula(ps, formula)

    otu_df = _otu_samples_rows(ps)

    shared = otu_df.index.intersection(x_df.index)
    return rda(otu_df.loc[shared], x_df.loc[shared], **kwargs)


# ---------------------------------------------------------------------------
# CAP — Constrained Analysis of Principal Coordinates
# ---------------------------------------------------------------------------


def _cap(
    ps: Phyloseq,
    distance: str | Any,
    formula: str | None,
    **kwargs: Any,
) -> Any:
    """Constrained Analysis of Principal Coordinates.

    Runs PCoA on the distance matrix, then applies RDA on the scores
    constrained by the formula variables.

    R reference: ordinate(physeq, "CAP", distance, formula)
    """
    from skbio.stats.ordination import rda

    dm = _resolve_dm(ps, distance)
    x_df = _parse_formula(ps, formula)

    pcoa_result = _pcoa(ps, dm)
    # Use PCoA sample scores as Y
    shared_ids = list(pcoa_result.samples.index.intersection(x_df.index))
    y = pcoa_result.samples.loc[shared_ids].astype(float)
    x = x_df.loc[shared_ids].astype(float)

    return rda(y, x, **kwargs)


# ---------------------------------------------------------------------------
# DPCoA ordination
# ---------------------------------------------------------------------------


def _dpcoa_ordinate(ps: Phyloseq, **kwargs: Any) -> Any:
    """Double Principal Coordinates Analysis ordination.

    R reference: ordinate(physeq, "DPCoA")
    """
    from pyloseq._distances import _dpcoa_manual  # noqa: PLC0415

    if ps.phy_tree is None:
        raise pyloseqValidationError("DPCoA ordination requires phy_tree")

    tree_node = ps.phy_tree._tree
    dm_species = tree_node.tip_tip_distances()

    otu_df = _otu_samples_rows(ps)

    common = [t for t in dm_species.ids if t in otu_df.columns]
    otu_df = otu_df[common]

    row_sums = otu_df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    freq_table = otu_df.div(row_sums, axis=0)

    return _dpcoa_manual(freq_table, dm_species)
