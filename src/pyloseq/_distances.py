"""Beta diversity distances, UniFrac, and the distance dispatcher."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, pdist, squareform
from skbio.diversity import beta_diversity
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import OrdinationResults

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

# Maps pyloseq/R method names -> (scipy_pdist_metric, binarize)
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
    "sorensen": ("dice", True),  # Dice = Sorensen
}

_PHYLO_METHODS = {"unifrac", "wunifrac"}
_SPECIAL_METHODS = {"jsd", "dpcoa"}

_ALL_METHODS = (
    sorted(_SCIPY_METHODS.keys()) + sorted(_PHYLO_METHODS) + sorted(_SPECIAL_METHODS)
)

# Keyword arguments that the UniFrac path legitimately accepts when reached
# through the generic distance() dispatcher. Anything else is rejected here
# with a clear message rather than surfacing as a TypeError from unifrac().
_UNIFRAC_ALLOWED_KWARGS = {"normalized", "n_jobs"}


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

    R reference: distance(physeq, method, type, ...)

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
        Passed to the underlying implementation. For UniFrac, only
        ``normalized`` and ``n_jobs`` are accepted. For scipy metrics, extra
        keywords are forwarded to :func:`scipy.spatial.distance.pdist` (e.g.
        ``p=`` for ``"minkowski"``, which otherwise defaults to ``p=2``).

    Returns
    -------
    skbio.stats.distance.DistanceMatrix
    """
    if "type" in kwargs:
        warnings.warn(
            "The 'type' parameter is deprecated; use 'kind' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kind = kwargs.pop("type")

    m = method.lower()

    if m in ("unifrac", "wunifrac"):
        bad = set(kwargs) - _UNIFRAC_ALLOWED_KWARGS
        if bad:
            raise pyloseqValidationError(
                f"Unsupported keyword(s) for {method!r}: {sorted(bad)}. "
                f"UniFrac accepts only {sorted(_UNIFRAC_ALLOWED_KWARGS)}."
            )
        return unifrac(ps, weighted=(m == "wunifrac"), **kwargs)
    if m == "jsd":
        return _jsd_distance(ps, kind=kind)
    if m == "dpcoa":
        return _dpcoa_distance(ps)

    if m in _SCIPY_METHODS:
        return _scipy_distance(ps, m, kind=kind, **kwargs)

    raise pyloseqValidationError(
        f"Unknown distance method: '{method}'. Supported: {_ALL_METHODS}"
    )


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

    R reference: UniFrac(physeq, weighted, normalized, parallel, fast)

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
    """

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
# Generalized UniFrac
# ---------------------------------------------------------------------------


def gunifrac(
    ps: Phyloseq,
    alpha: tuple[float, ...] = (0, 0.5, 1),
) -> dict[str, DistanceMatrix]:
    """Compute Generalized UniFrac distance matrices.

    Implements the GUniFrac family from Chen et al. (2012) *Bioinformatics*
    28(16):2106-2113, matching the R ``GUniFrac`` package API.

    R reference: GUniFrac(otu.tab, tree, alpha=c(0, 0.5, 1))$unifracs

    Parameters
    ----------
    ps:
        ``Phyloseq`` object with both ``otu_table`` and ``phy_tree``.
    alpha:
        Exponents to compute. Each value in ``[0, 1]`` produces one distance
        matrix keyed ``"d_{alpha}"`` (e.g. ``"d_0.5"``). Alpha = 0 gives the
        same result as unweighted UniFrac; alpha = 1 gives (un-normalised)
        weighted UniFrac.

    Returns
    -------
    dict mapping:

    - ``"d_{a}"`` for each ``a`` in *alpha* — GUniFrac at that exponent.
      ``d_0`` equals ``d_UW``; ``d_1`` equals normalized weighted UniFrac.
    - ``"d_UW"``  — unweighted UniFrac as defined in R's GUniFrac package
      (fraction of branches where cumulative proportions differ; see note below)
    - ``"d_VAW"`` — variance-adjusted weighted UniFrac (Hamady et al. 2010)

    .. note::
        ``d_UW`` from this function matches R's ``GUniFrac`` package definition,
        which counts any branch whose cumulative proportion differs between the
        two samples (including branches shared by both but in different amounts).
        This differs slightly from scikit-bio / :func:`unifrac` which counts only
        branches exclusive to one sample (the Lozupone & Knight 2005 definition).
    """
    if ps.phy_tree is None:
        raise pyloseqValidationError("gunifrac requires phy_tree")

    tree = ps.phy_tree._tree
    tree_tips = {n.name for n in tree.tips()}

    otu_df = _otu_samples_rows(ps)
    taxa_in_tree = [t for t in otu_df.columns if t in tree_tips]
    if not taxa_in_tree:
        raise pyloseqValidationError(
            "No taxa names match tree tip labels. "
            "Check that taxa_names and tree tip names are consistent."
        )
    otu_df = otu_df[taxa_in_tree]

    row_sums = otu_df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    prop_df = otu_df.div(row_sums, axis=0)

    sample_ids = list(prop_df.index)
    n = len(sample_ids)
    prop_mat = prop_df.values.astype(float)  # (n_samples, n_taxa)
    tip_to_col: dict[str, int] = {t: i for i, t in enumerate(prop_df.columns)}

    # Post-order traversal: accumulate cumulative proportions per branch.
    # cp[b, s] = sum of sample s's relative abundances in the subtree above
    # branch b.  bl[b] = length of branch b.
    branch_lengths: list[float] = []
    cum_props_list: list[np.ndarray] = []
    node_props: dict[object, np.ndarray] = {}

    for node in tree.postorder():
        if node.is_tip():
            name = node.name
            if name in tip_to_col:
                node_props[node] = prop_mat[:, tip_to_col[name]].copy()
            else:
                node_props[node] = np.zeros(n)
        else:
            node_props[node] = np.sum([node_props[c] for c in node.children], axis=0)

        if node.length is not None and node.parent is not None and node.length > 0:
            branch_lengths.append(float(node.length))
            cum_props_list.append(node_props[node].copy())

    bl = np.array(branch_lengths, dtype=float)  # (n_branches,)
    cp = np.array(cum_props_list, dtype=float)  # (n_branches, n_samples)

    results: dict[str, DistanceMatrix] = {}

    def _make_dm(data: np.ndarray) -> DistanceMatrix:
        return DistanceMatrix(data, ids=sample_ids)

    # --- GUniFrac d_alpha ---
    # Formula from R's GUniFrac package (Chen et al. 2012):
    #   diff = |p_bi - p_bj| / (p_bi + p_bj)      (relative difference, ∈ [0,1])
    #   w    = l_b * (p_bi + p_bj)^alpha            (branch weight)
    #   d_alpha(i,j) = sum(diff * w) / sum(w)
    #                = sum(l_b * |p1-p2| * (p1+p2)^(alpha-1)) / sum(l_b * (p1+p2)^alpha)
    # Valid for alpha >= 0.  alpha=1 reduces to weighted UniFrac.
    # alpha=0: diff/s is bounded in [0,1] since |p1-p2| <= p1+p2, so s^(-1) is safe.
    for a in alpha:
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                pi = cp[:, i]
                pj = cp[:, j]
                mask = (pi + pj) > 0
                if not mask.any():
                    continue
                bm = bl[mask]
                diff = np.abs(pi[mask] - pj[mask])
                s = pi[mask] + pj[mask]
                num = np.dot(bm, diff * s ** (a - 1))
                den = np.dot(bm, s**a)
                dm[i, j] = dm[j, i] = num / den if den > 0 else 0.0
        results[f"d_{a}"] = _make_dm(dm)

    # --- Unweighted UniFrac (d_UW) ---
    # Standard unweighted UniFrac: binarize cumulative proportions first, then
    # compute the fraction of shared branch length that is unique to one sample.
    # Numerator = branch length present in exactly one sample (XOR of presence).
    # Denominator = branch length present in at least one sample.
    dm_uw = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            I1 = cp[:, i] > 0
            I2 = cp[:, j] > 0
            either = I1 | I2
            unique = I1 ^ I2
            den = bl[either].sum()
            v = bl[unique].sum() / den if den > 0 else 0.0
            dm_uw[i, j] = dm_uw[j, i] = v
    results["d_UW"] = _make_dm(dm_uw)

    # --- Variance-adjusted weighted UniFrac (d_VAW) ---
    # d_VAW(i,j) = sqrt( sum_b b*(p_bi - p_bj)^2/(p_bi+p_bj) )
    #            / sqrt( sum_b b*(p_bi + p_bj) )
    dm_vaw = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pi = cp[:, i]
            pj = cp[:, j]
            s = pi + pj
            mask = s > 0
            if not mask.any():
                continue
            bm = bl[mask]
            diff = pi[mask] - pj[mask]
            sm = s[mask]
            num = np.dot(bm, diff**2 / sm)
            den = np.dot(bm, sm)
            dm_vaw[i, j] = dm_vaw[j, i] = np.sqrt(num / den) if den > 0 else 0.0
    results["d_VAW"] = _make_dm(dm_vaw)

    return results


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
        otu_df = otu_df.T  # -> taxa x samples

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

    otu_df = _otu_samples_rows(ps)
    if kind == "taxa":
        otu_df = otu_df.T

    mat = otu_df.values.astype(float)
    # Normalise rows to sum to 1 (probability distributions)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    ids = list(otu_df.index)
    # base=2 ensures output is in [0, 1]: JSD in [0, 1] with base-2 log,
    # and scipy.jensenshannon returns sqrt(JSD), so the distance in [0, 1].
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

    species_ids = list(dm_species.ids)
    D = np.array(dm_species.data)

    common = [t for t in species_ids if t in freq_table.columns]
    if not common:
        raise pyloseqValidationError(
            "No shared taxa between frequency table and species distance matrix"
        )

    W = freq_table[common].values.astype(float)  # n x p
    common_idx = [species_ids.index(t) for t in common]
    D_sub = D[np.ix_(common_idx, common_idx)]
    p = len(common)

    # Double-center D^2
    D2 = D_sub**2
    H = np.eye(p) - np.ones((p, p)) / p
    Q = -0.5 * H @ D2 @ H  # p x p species inner-product matrix

    # Inter-sample inner-product matrix
    S = W @ Q @ W.T  # n x n
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
