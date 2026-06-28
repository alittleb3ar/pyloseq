"""Alpha diversity estimation, mirroring R phyloseq/vegan's estimate_richness."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pyloseq._exceptions import pyloseqValidationError
from pyloseq._manipulation import _otu_samples_rows

if TYPE_CHECKING:
    from pyloseq._phyloseq import Phyloseq

# Chao & Lee 1992: taxa with counts <= this threshold are considered "rare"
_ACE_RARE_THRESHOLD: int = 10

_ALL_MEASURES = [
    "Observed",
    "Chao1",
    "se.chao1",
    "ACE",
    "Shannon",
    "Simpson",
    "InvSimpson",
    "Fisher",
    "PD",
]

# Measures that require a rooted phylogenetic tree
_TREE_MEASURES = {"PD"}


def estimate_richness(
    ps: Phyloseq,
    measures: list[str] | None = None,
    split: bool = True,
) -> pd.DataFrame:
    """Estimate richness (alpha diversity) for each sample.

    R reference: estimate_richness(physeq, split, measures)

    Parameters
    ----------
    ps:
        ``Phyloseq`` object. OTU table values should be integer counts.
    measures:
        Subset of ``["Observed", "Chao1", "se.chao1", "ACE", "se.ACE",
        "Shannon", "Simpson", "InvSimpson", "Fisher", "PD"]``.
        ``None`` (default) returns all non-tree measures (``PD`` is excluded
        from the default set when the object has no ``phy_tree``).
    split:
        If ``True`` (default), compute per sample. If ``False``, pool all
        samples first (matches R behavior). The single pooled row is labelled
        ``"pooled"``.

    Returns
    -------
    pd.DataFrame
        Indexed by sample name (or ``"pooled"`` when ``split=False``); columns
        are the requested measures.

    Raises
    ------
    pyloseqValidationError
        If ``"PD"`` is requested but ``ps.phy_tree`` is ``None``, or if any
        measure name is not recognised.

    Notes
    -----
    ``"PD"`` (Faith's phylogenetic diversity, Faith 1992) sums the total branch
    length of the minimum spanning clade connecting all observed taxa on the
    tree.  It requires a rooted tree; the tree is midpoint-rooted internally if
    not already rooted, matching R's ``phangorn::midpoint`` convention.
    """
    if measures is None:
        # Exclude PD from the default set when there is no tree to avoid
        # surprising errors for callers who don't know about PD.
        if ps.phy_tree is not None:
            measures = list(_ALL_MEASURES)
        else:
            measures = [m for m in _ALL_MEASURES if m not in _TREE_MEASURES]
    else:
        bad = [m for m in measures if m not in _ALL_MEASURES]
        if bad:
            raise pyloseqValidationError(
                f"Unknown measure(s): {bad}. Choose from: {_ALL_MEASURES}"
            )

    if "PD" in measures and ps.phy_tree is None:
        raise pyloseqValidationError(
            "'PD' (Faith's phylogenetic diversity) requires a phylogenetic tree "
            "(phy_tree) on the Phyloseq object."
        )

    if "se.ACE" in measures:
        warnings.warn(
            "se.ACE is not implemented and will always be NaN. Use ACE for the point estimate.",
            UserWarning,
            stacklevel=2,
        )

    non_tree = [m for m in measures if m not in _TREE_MEASURES]
    otu_df = _otu_samples_rows(ps)  # samples × taxa

    if not split:
        # Pool across all samples: sum each taxon's counts, compute one row.
        pooled = otu_df.sum(axis=0)
        rows = {"pooled": _richness_single(np.asarray(pooled), non_tree)}
    else:
        rows = {
            str(sid): _richness_single(np.asarray(row), non_tree)
            for sid, row in otu_df.iterrows()
        }

    df = pd.DataFrame.from_dict(rows, orient="index")

    if "PD" in measures:
        df["PD"] = _faith_pd(ps, otu_df, split=split)

    return df[measures]


def _faith_pd(ps: Phyloseq, otu_df: pd.DataFrame, split: bool) -> pd.Series:
    """Compute Faith's phylogenetic diversity (Faith 1992) via scikit-bio.

    Midpoint-roots the tree if not already rooted, matching R's phangorn::midpoint
    convention.  Taxa absent from the tree are silently dropped before computing.
    """
    from skbio.diversity import alpha_diversity  # noqa: PLC0415

    tree_node = ps.phy_tree._tree
    if not ps.phy_tree.is_rooted:
        tree_node = tree_node.root_at_midpoint()

    tree_tips = {n.name for n in tree_node.tips()}
    common_taxa = [t for t in otu_df.columns if t in tree_tips]

    if not common_taxa:
        raise pyloseqValidationError(
            "No OTU/ASV names in the OTU table match tree tip labels. "
            "Verify that taxa names are consistent between the OTU table and tree."
        )

    if not split:
        pooled = otu_df.sum(axis=0)
        mat = pooled[common_taxa].values.astype(int).reshape(1, -1)
        return alpha_diversity(
            "faith_pd",
            mat,
            ids=["pooled"],
            tree=tree_node,
            taxa=common_taxa,
        )

    aligned = otu_df[common_taxa].astype(int)
    return alpha_diversity(
        "faith_pd",
        aligned.values,
        ids=list(aligned.index),
        tree=tree_node,
        taxa=common_taxa,
    )


# ---------------------------------------------------------------------------
# Single-sample helpers
# ---------------------------------------------------------------------------


def _richness_single(raw: np.ndarray, measures: list[str]) -> dict[str, float]:
    """Compute richness measures for one sample's count vector."""
    counts: np.ndarray = raw.astype(float)
    counts_int = np.round(counts).astype(int)
    nonzero = counts_int[counts_int > 0]

    n = int(nonzero.sum())
    s_obs = len(nonzero)

    row: dict[str, float] = {}

    if "Observed" in measures:
        row["Observed"] = float(s_obs)

    if any(m in measures for m in ("Chao1", "se.chao1")):
        c1, se1 = _chao1(nonzero)
        row["Chao1"] = c1
        row["se.chao1"] = se1

    if any(m in measures for m in ("ACE", "se.ACE")):
        ace, se_ace = _ace(nonzero)
        row["ACE"] = ace
        row["se.ACE"] = se_ace

    if "Shannon" in measures:
        if n > 0:
            p = nonzero / n
            row["Shannon"] = float(-np.sum(p * np.log(p)))
        else:
            row["Shannon"] = float("nan")

    if "Simpson" in measures or "InvSimpson" in measures:
        if n > 0:
            p = nonzero / n
            d = float(np.sum(p**2))
            row["Simpson"] = 1.0 - d
            row["InvSimpson"] = 1.0 / d if d > 0 else float("inf")
        else:
            row["Simpson"] = float("nan")
            row["InvSimpson"] = float("nan")

    if "Fisher" in measures:
        row["Fisher"] = _fisher_alpha(n, s_obs)

    return row


def _chao1(nonzero: np.ndarray) -> tuple[float, float]:
    """Chao1 estimator and its standard error (Colwell et al. 2004)."""
    s_obs = len(nonzero)
    f1 = int(np.sum(nonzero == 1))
    f2 = int(np.sum(nonzero == 2))

    if f2 > 0:
        chao1 = s_obs + f1**2 / (2.0 * f2)
        r = f1 / f2
        se = float(np.sqrt(f2 * (r**4 / 4.0 + r**3 + r**2 / 2.0)))
    else:
        # f2 == 0: bias-corrected Chao1 and its variance (Colwell 2004, eq. for
        # the f2 == 0 case). Var = f1(f1-1)/2 + f1(2 f1 - 1)^2 / 4 - f1^4 / (4 S),
        # where S is the (bias-corrected) Chao1 estimate.
        chao1 = s_obs + f1 * (f1 - 1) / 2.0
        if f1 == 0:
            se = 0.0
        else:
            var = (
                f1 * (f1 - 1) / 2.0
                + f1 * (2 * f1 - 1) ** 2 / 4.0
                - f1**4 / (4.0 * chao1)
            )
            se = float(np.sqrt(var)) if var > 0 else 0.0

    return chao1, se


def _ace(nonzero: np.ndarray) -> tuple[float, float]:
    """Abundance Coverage Estimator (Chao & Lee 1992; matches vegan::estimateR)."""
    s_obs = len(nonzero)
    threshold = _ACE_RARE_THRESHOLD
    rare = nonzero[nonzero <= threshold]
    abund = nonzero[nonzero > threshold]

    s_rare = len(rare)
    s_abund = len(abund)
    n_rare = int(rare.sum())

    if n_rare == 0 or s_rare == 0:
        return float(s_obs), float("nan")

    f_i = np.array([int(np.sum(rare == i)) for i in range(1, threshold + 1)])
    f1 = f_i[0]

    c_ace = 1.0 - f1 / n_rare  # coverage estimate

    if c_ace <= 0:
        return float(s_obs), float("nan")

    # gamma^2_ace
    denom = n_rare * (n_rare - 1)
    if denom <= 0:
        gamma_sq = 0.0
    else:
        gamma_sq = max(
            s_rare
            / c_ace
            * sum(int(i + 1) * int(i) * int(f_i[i]) for i in range(len(f_i)))
            / denom
            - 1.0,
            0.0,
        )

    ace = s_abund + s_rare / c_ace + f1 / c_ace * gamma_sq
    return ace, float("nan")


def _fisher_alpha(n: int, s_obs: int) -> float:
    """Fisher's log-series alpha via Brent root-finding (matches R vegan::fisher.alpha)."""

    if s_obs == 0 or n == 0:
        return float("nan")

    def eq(a: float) -> float:
        return float(a * np.log(1.0 + n / a) - s_obs)

    # Find a bracket: eq(small) < 0 when alpha is tiny, grows with alpha
    try:
        # Upper bound: alpha * ln(n/alpha) ~ s_obs  ->  alpha ~ s_obs / ln(n)
        upper = max(float(n) * 10, float(s_obs) * 100)
        return float(brentq(eq, 1e-8, upper, xtol=1e-12, rtol=1e-12))
    except ValueError:
        return float("nan")
