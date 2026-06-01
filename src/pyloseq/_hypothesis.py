"""Multiple hypothesis testing for differential abundance analysis.

Implements per-taxon tests for comparing two groups, with several
multiple-testing correction options including the permutation-based
Westfall-Young step-down FWER procedure that R's multtest::mt.minP uses.

R reference: phyloseq::mt(), multtest::mt.maxT, multtest::mt.minP
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyloseq._exceptions import pyloseqValidationError

if TYPE_CHECKING:
    from pyloseq._phyloseq import Phyloseq


def multi_tax_test(
    ps: Phyloseq,
    grouping_var: str,
    test: Literal["t", "wilcoxon"] = "t",
    method: Literal["BH", "BY", "holm", "bonferroni", "westfall_young"] = "BH",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    n_permutations: int = 1000,
    rng_seed: int | None = 42,
) -> pd.DataFrame:
    """Test each taxon for differential abundance between two groups.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object (must have ``sample_data``).
    grouping_var:
        Column in ``sample_data`` defining the two groups to compare.
        Samples with ``NaN`` in this column are silently dropped.
    test:
        Per-taxon test statistic. ``"t"`` uses Welch's t-test
        (``equal_var=False``); ``"wilcoxon"`` uses the Wilcoxon rank-sum test.
    method:
        Multiple-testing correction method:

        - ``"BH"``            — Benjamini-Hochberg FDR (default)
        - ``"BY"``            — Benjamini-Yekutieli FDR
        - ``"holm"``          — Holm step-down FWER
        - ``"bonferroni"``    — Bonferroni FWER
        - ``"westfall_young"`` — permutation-based step-down FWER
          (R ``multtest::mt.minP``)
    alternative:
        Direction of the alternative hypothesis.
    n_permutations:
        Number of label permutations for ``method="westfall_young"``.
    rng_seed:
        Seed for the permutation RNG (``"westfall_young"`` only).
        Pass ``None`` for non-reproducible draws.

    Returns
    -------
    pd.DataFrame
        One row per taxon, sorted by ascending ``adjp``. Columns:

        - ``statistic``     — per-taxon test statistic
        - ``rawp``          — uncorrected p-value
        - ``adjp``          — corrected p-value (using ``method``)
        - ``mean_<g1>``     — mean abundance in group 1
        - ``mean_<g2>``     — mean abundance in group 2

    Raises
    ------
    pyloseqValidationError
        If ``sample_data`` is missing, ``grouping_var`` is not found, the
        variable does not have exactly 2 distinct non-NaN levels, or either
        group has fewer than 2 samples.

    R reference: phyloseq::mt()
    """
    if ps.sample_data is None:
        raise pyloseqValidationError("multi_tax_test requires sample_data")

    from pyloseq._manipulation import _otu_taxa_rows  # noqa: PLC0415

    sam_df = ps.sample_data.to_frame()
    if grouping_var not in sam_df.columns:
        raise pyloseqValidationError(
            f"grouping_var '{grouping_var}' not found in sample_data. "
            f"Available: {list(sam_df.columns)}"
        )

    groups = sam_df[grouping_var].dropna()
    unique_groups = sorted(groups.unique(), key=str)
    if len(unique_groups) != 2:
        raise pyloseqValidationError(
            f"multi_tax_test requires exactly 2 non-NaN groups in '{grouping_var}'; "
            f"found {len(unique_groups)}: {list(unique_groups)}"
        )
    g1_label, g2_label = str(unique_groups[0]), str(unique_groups[1])

    g1_samples = list(groups[groups == unique_groups[0]].index)
    g2_samples = list(groups[groups == unique_groups[1]].index)

    if len(g1_samples) < 2 or len(g2_samples) < 2:
        raise pyloseqValidationError(
            f"Each group must have at least 2 samples. "
            f"Got {len(g1_samples)} for '{g1_label}', {len(g2_samples)} for '{g2_label}'."
        )

    otu_df = _otu_taxa_rows(ps).reindex(columns=g1_samples + g2_samples)

    a = otu_df[g1_samples].values.astype(float)  # (M, n1)
    b = otu_df[g2_samples].values.astype(float)  # (M, n2)

    stats, rawp = _test_all_taxa(a, b, test=test, alternative=alternative)

    # Constant taxa (e.g. all-zero) produce NaN; treat as non-significant
    rawp = np.where(np.isnan(rawp), 1.0, rawp)
    stats = np.where(np.isnan(stats), 0.0, stats)

    if method == "westfall_young":
        adjp = _westfall_young_fwer(
            a,
            b,
            rawp,
            test=test,
            n_perm=n_permutations,
            rng_seed=rng_seed,
        )
    else:
        adjp = _adjust_pvalues(rawp, method=method)

    return pd.DataFrame(
        {
            "statistic": stats,
            "rawp": rawp,
            "adjp": adjp,
            f"mean_{g1_label}": a.mean(axis=1),
            f"mean_{g2_label}": b.mean(axis=1),
        },
        index=otu_df.index,
    ).sort_values("adjp")


# ---------------------------------------------------------------------------
# Internal: per-taxon test statistics
# ---------------------------------------------------------------------------


def _test_all_taxa(
    a: np.ndarray,
    b: np.ndarray,
    test: str,
    alternative: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(statistics, pvalues)`` arrays of shape ``(M,)``."""
    if test == "t":
        from scipy.stats import ttest_ind  # noqa: PLC0415

        stats, pvals = ttest_ind(a, b, axis=1, equal_var=False, alternative=alternative)
        return np.asarray(stats, dtype=float), np.asarray(pvals, dtype=float)

    if test == "wilcoxon":
        from scipy.stats import ranksums  # noqa: PLC0415

        M = a.shape[0]
        stats = np.zeros(M)
        pvals = np.zeros(M)
        for i in range(M):
            try:
                stats[i], pvals[i] = ranksums(a[i], b[i], alternative=alternative)
            except ValueError:
                stats[i], pvals[i] = 0.0, 1.0
        return stats, pvals

    raise pyloseqValidationError(f"Unknown test '{test}'. Use 't' or 'wilcoxon'.")


# ---------------------------------------------------------------------------
# Internal: p-value adjustment
# ---------------------------------------------------------------------------


def _adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    """Apply a multiple-testing correction to a 1-D array of p-values."""
    if method in ("BH", "BY"):
        from scipy.stats import false_discovery_control  # noqa: PLC0415

        return np.asarray(
            false_discovery_control(pvals, method=method.lower()),
            dtype=float,
        )

    if method == "bonferroni":
        return np.minimum(pvals * len(pvals), 1.0)

    if method == "holm":
        return _holm(pvals)

    raise pyloseqValidationError(
        f"Unknown method '{method}'. "
        "Use 'BH', 'BY', 'holm', 'bonferroni', or 'westfall_young'."
    )


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down FWER correction (Holm 1979)."""
    M = len(pvals)
    order = np.argsort(pvals)
    # Multiply each sorted p by its step-down factor (M, M-1, ..., 1)
    adjusted = pvals[order] * np.arange(M, 0, -1)
    # Monotonicity: each adjusted p must be >= the previous
    adjusted = np.maximum.accumulate(adjusted)
    result = np.empty(M)
    result[order] = np.minimum(adjusted, 1.0)
    return result


# ---------------------------------------------------------------------------
# Internal: Westfall-Young step-down FWER (mt.minP)
# ---------------------------------------------------------------------------


def _westfall_young_fwer(
    a: np.ndarray,
    b: np.ndarray,
    obs_pvals: np.ndarray,
    test: str,
    n_perm: int,
    rng_seed: int | None,
) -> np.ndarray:
    """Westfall-Young step-down minP permutation FWER.

    Procedure (Westfall & Young 1993, Algorithm 2.8):
    1. Sort taxa by ascending observed p-value (most significant first).
    2. For each permutation, permute group labels and compute per-taxon
       p-values; then compute the successive *minimum* p-value running
       from the least-significant to the most-significant taxon.
    3. The adjusted p-value for taxon j is the fraction of permutations
       in which that running minimum was <= the observed p-value at rank j.
    4. Enforce monotonicity: adjusted p-values must be non-decreasing
       from most to least significant.

    R reference: multtest::mt.minP
    """
    rng = np.random.default_rng(rng_seed)
    M = len(obs_pvals)
    n1 = a.shape[1]
    ab = np.concatenate([a, b], axis=1)  # (M, N)
    N = ab.shape[1]

    order = np.argsort(obs_pvals)  # most significant first
    sorted_obs = obs_pvals[order]

    exceed = np.zeros(M, dtype=np.intp)

    for _ in range(n_perm):
        perm = rng.permutation(N)
        _, perm_pvals = _test_all_taxa(
            ab[:, perm[:n1]], ab[:, perm[n1:]], test=test, alternative="two-sided"
        )
        perm_pvals = np.where(np.isnan(perm_pvals), 1.0, perm_pvals)
        perm_sorted = perm_pvals[order]
        # Successive minima from right (least significant) to left
        cum_min = np.minimum.accumulate(perm_sorted[::-1])[::-1]
        exceed += cum_min <= sorted_obs

    raw_adj = exceed / n_perm
    # Monotonicity: non-decreasing from most to least significant
    monotone = np.maximum.accumulate(raw_adj)
    result = np.empty(M)
    result[order] = np.minimum(monotone, 1.0)
    return result
