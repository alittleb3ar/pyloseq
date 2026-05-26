"""Beta diversity distances, UniFrac, and the distance dispatcher.

R reference: phyloseq::distance(physeq, method, type, ...)
             phyloseq::UniFrac(physeq, weighted, normalized, parallel, fast)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyloseq._exceptions import pyloseqValidationError
from pyloseq._manipulation import _otu_samples_rows

if TYPE_CHECKING:
    from skbio.stats.distance import DistanceMatrix

    from pyloseq._phyloseq import Phyloseq

# Threshold for treating near-zero eigenvalues as positive in DPCoA
_PCOA_EIGENVALUE_FLOOR: float = 1e-10

# ---------------------------------------------------------------------------
# Method catalogue
# ---------------------------------------------------------------------------

# Maps pyloseq/R method names → (scipy_pdist_metric, binarize)
_SCIPY_METHODS: dict[str, tuple[str, bool]] = {
    "euclidean": ("euclidean", False),
    "manhattan": ("cityblock", False),
    "canberra": ("canberra", False),
    "bray": ("braycurtis", False),
    "jaccard": ("jaccard", True),  # phyloseq uses binary Jaccard by default
    "binary": ("jaccard", True),
    "maximum": ("chebyshev", False),
    "minkowski": ("minkowski", False),
    "cosine": ("cosine", False),
    "correlation": ("correlation", False),
    "sorensen": ("dice", True),  # Dice = Sørensen
}

_PHYLO_METHODS = {"unifrac", "wunifrac"}
_SPECIAL_METHODS = {"jsd", "dpcoa"}

_ALL_METHODS = sorted(_SCIPY_METHODS.keys()) + sorted(_PHYLO_METHODS) + sorted(_SPECIAL_METHODS)


def distance_method_list() -> dict[str, list[str]]:
    """Return all supported distance methods, grouped by backend.

    R reference: distanceMethodList
    """
    return {
        "phylogenetic": sorted(_PHYLO_METHODS | {"dpcoa"}),
        "information": ["jsd"],
        "vegan-equivalent": sorted(_SCIPY_METHODS.keys()),
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def distance(
    ps: Phyloseq,
    method: str,
    kind: str = "samples",
    **kwargs: Any,
) -> DistanceMatrix:
    """Compute a pairwise distance (or dissimilarity) matrix.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    method:
        Distance method.  See :func:`distance_method_list` for all options.
    kind:
        ``"samples"`` (default) or ``"taxa"``.  Most phylogenetic methods
        require ``"samples"``.
    **kwargs:
        Passed to the underlying implementation (e.g. ``weighted`` for
        UniFrac).

    Returns
    -------
    skbio.stats.distance.DistanceMatrix

    R reference: distance(physeq, method, type, ...)
    """
    if "type" in kwargs:
        warnings.warn(
            "The 'type' parameter is deprecated; use 'kind' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kind = kwargs.pop("type")

    m = method.lower()

    if m == "unifrac":
        return unifrac(ps, weighted=False, **kwargs)
    if m == "wunifrac":
        return unifrac(ps, weighted=True, **kwargs)
    if m == "jsd":
        return _jsd_distance(ps, kind=kind)
    if m == "dpcoa":
        return _dpcoa_distance(ps)

    if m in _SCIPY_METHODS:
        return _scipy_distance(ps, m, kind=kind, **kwargs)

    raise pyloseqValidationError(f"Unknown distance method: '{method}'. Supported: {_ALL_METHODS}")


# ---------------------------------------------------------------------------
# UniFrac
# ---------------------------------------------------------------------------


def unifrac(
    ps: Phyloseq,
    weighted: bool = False,
    normalized: bool = True,
    n_jobs: int = 1,
) -> DistanceMatrix:
    """Compute (weighted or unweighted) UniFrac distances.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object with both ``otu_table`` and ``phy_tree``.
    weighted:
        If ``True``, compute weighted UniFrac; otherwise unweighted.
    normalized:
        Normalize by total branch length (meaningful only for weighted UniFrac).
    n_jobs:
        Number of parallel workers (passed to scikit-bio).

    Returns
    -------
    skbio.stats.distance.DistanceMatrix

    R reference: UniFrac(physeq, weighted, normalized, parallel, fast)
    """
    from skbio.diversity import beta_diversity

    if ps.phy_tree is None:
        raise pyloseqValidationError("unifrac requires phy_tree")

    tree_node = ps.phy_tree._tree
    tree_tips = set(ps.phy_tree.tip_names)

    otu_df = _otu_samples_rows(ps)

    # Restrict to taxa present in the tree
    taxa_in_tree = [t for t in otu_df.columns if t in tree_tips]
    if not taxa_in_tree:
        raise pyloseqValidationError(
            "No taxa names match tree tip labels. "
            "Check that taxa_names and tree tip names are consistent."
        )
    otu_df = otu_df[taxa_in_tree]

    counts = otu_df.values.astype(int)
    sample_ids = list(otu_df.index)
    otu_ids = list(otu_df.columns)

    metric = "weighted_unifrac" if weighted else "unweighted_unifrac"

    kwargs: dict[str, Any] = {
        "tree": tree_node,
        "taxa": otu_ids,
    }
    if weighted:
        kwargs["normalized"] = normalized

    dm = beta_diversity(metric, counts, sample_ids, **kwargs)
    return dm


# ---------------------------------------------------------------------------
# Non-phylogenetic distances (scipy-backed)
# ---------------------------------------------------------------------------


def _scipy_distance(
    ps: Phyloseq,
    method: str,
    kind: str = "samples",
    **kwargs: Any,
) -> Any:
    """Compute pairwise distance matrix via scipy.spatial.distance.pdist."""
    from scipy.spatial.distance import pdist, squareform
    from skbio.stats.distance import DistanceMatrix

    scipy_metric, binarize = _SCIPY_METHODS[method]

    otu_df = _otu_samples_rows(ps)

    if kind == "taxa":
        otu_df = otu_df.T  # → taxa × samples

    mat = otu_df.values.astype(float)
    if binarize:
        mat = (mat > 0).astype(float)

    ids = list(otu_df.index)
    condensed = pdist(mat, metric=scipy_metric, **kwargs)
    sq = squareform(condensed)
    return DistanceMatrix(sq, ids=ids)


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------


def _jsd_distance(ps: Phyloseq, kind: str = "samples") -> Any:
    """Pairwise Jensen-Shannon divergence."""
    from scipy.spatial.distance import jensenshannon, pdist, squareform
    from skbio.stats.distance import DistanceMatrix

    otu_df = _otu_samples_rows(ps)
    if kind == "taxa":
        otu_df = otu_df.T

    mat = otu_df.values.astype(float)
    # Normalise rows to sum to 1 (probability distributions)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    ids = list(otu_df.index)
    # base=2 ensures output is in [0, 1]: JSD ∈ [0, 1] with base-2 log,
    # and scipy.jensenshannon returns sqrt(JSD), so the distance ∈ [0, 1].
    condensed = pdist(mat, metric=lambda u, v: jensenshannon(u, v, base=2))
    sq = squareform(condensed)
    return DistanceMatrix(sq, ids=ids)


# ---------------------------------------------------------------------------
# Double Principal Coordinates Analysis (DPCoA)
# ---------------------------------------------------------------------------


def _dpcoa_manual(freq_table: pd.DataFrame, dm_species: Any) -> Any:
    """DPCoA implementation (Pavoine et al. 2004) returning OrdinationResults.

    scikit-bio 0.6+ removed the standalone dpcoa() function, so we implement
    the double-centering + weighted projection algorithm directly.
    """
    from skbio.stats.ordination import OrdinationResults

    species_ids = list(dm_species.ids)
    D = np.array(dm_species.data)

    common = [t for t in species_ids if t in freq_table.columns]
    if not common:
        raise pyloseqValidationError(
            "No shared taxa between frequency table and species distance matrix"
        )

    W = freq_table[common].values.astype(float)  # n × p
    common_idx = [species_ids.index(t) for t in common]
    D_sub = D[np.ix_(common_idx, common_idx)]
    p = len(common)

    # Double-center D²
    D2 = D_sub**2
    H = np.eye(p) - np.ones((p, p)) / p
    Q = -0.5 * H @ D2 @ H  # p × p species inner-product matrix

    # Inter-sample inner-product matrix
    S = W @ Q @ W.T  # n × n
    S = (S + S.T) / 2  # symmetrise numerical noise

    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]

    pos = vals > _PCOA_EIGENVALUE_FLOOR
    n_pos = int(pos.sum())
    vals_pos = vals[:n_pos]
    coords = vecs[:, :n_pos] * np.sqrt(vals_pos)

    sample_ids = list(freq_table.index)
    col_names = [f"DPCoA{i + 1}" for i in range(n_pos)]
    total_var = float(vals_pos.sum())

    return OrdinationResults(
        short_method_name="DPCoA",
        long_method_name="Double Principal Coordinates Analysis",
        eigvals=pd.Series(vals_pos, index=col_names),
        samples=pd.DataFrame(coords, index=sample_ids, columns=col_names),
        proportion_explained=pd.Series(
            vals_pos / total_var if total_var > 0 else np.zeros(n_pos),
            index=col_names,
        ),
    )


def _dpcoa_distance(ps: Phyloseq) -> Any:
    """Compute sample distances via Double PCoA on patristic distances.

    Implements the DPCoA metric described by Pavoine et al. 2004.

    R reference: distance(physeq, "dpcoa")
    """
    from skbio.stats.distance import DistanceMatrix

    if ps.phy_tree is None:
        raise pyloseqValidationError("dpcoa requires phy_tree")

    tree_node = ps.phy_tree._tree
    dm_species = tree_node.tip_tip_distances()

    otu_df = _otu_samples_rows(ps)

    common = [t for t in dm_species.ids if t in otu_df.columns]
    otu_df = otu_df[common]

    row_sums = otu_df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    freq_table = otu_df.div(row_sums, axis=0)

    result = _dpcoa_manual(freq_table, dm_species)

    from scipy.spatial.distance import pdist, squareform  # noqa: PLC0415

    sample_coords = result.samples.values
    sq = squareform(pdist(sample_coords))
    return DistanceMatrix(sq, ids=list(result.samples.index))
