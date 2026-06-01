"""Alpha diversity estimation, mirroring R phyloseq/vegan's estimate_richness.

R reference: phyloseq::estimate_richness(physeq, split, measures)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
    "se.ACE",
    "Shannon",
    "Simpson",
    "InvSimpson",
    "Fisher",
]


def estimate_richness(
    ps: Phyloseq,
    measures: list[str] | None = None,
    split: bool = True,
) -> pd.DataFrame:
    """Estimate richness (alpha diversity) for each sample.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object. OTU table values should be integer counts.
    measures:
        Subset of ``["Observed", "Chao1", "se.chao1", "ACE", "se.ACE",
        "Shannon", "Simpson", "InvSimpson", "Fisher"]``.
        ``None`` (default) returns all.
    split:
        If ``True`` (default), compute per sample. If ``False``, pool all
        samples first (matches R behavior). The single pooled row is labelled
        ``"pooled"``.

    Returns
    -------
    pd.DataFrame
        Indexed by sample name (or ``"pooled"`` when ``split=False``); columns
        are the requested measures.

    R reference: estimate_richness(physeq, split, measures)
    """
    if measures is None:
        measures = list(_ALL_MEASURES)
    else:
        bad = [m for m in measures if m not in _ALL_MEASURES]
        if bad:
            raise pyloseqValidationError(
                f"Unknown measure(s): {bad}. Choose from: {_ALL_MEASURES}"
            )

    if "se.ACE" in measures:
        warnings.warn(
            "se.ACE is not implemented and will always be NaN. Use ACE for the point estimate.",
            UserWarning,
            stacklevel=2,
        )

    otu_df = _otu_samples_rows(ps)

    if not split:
        # Pool across all samples: sum each taxon's counts, compute one row.
        pooled = otu_df.sum(axis=0)
        rows = {"pooled": _richness_single(np.asarray(pooled), measures)}
    else:
        rows = {
            str(sid): _richness_single(np.asarray(row), measures)
            for sid, row in otu_df.iterrows()
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df[measures]


# ---------------------------------------------------------------------------
# Single-sample helpers
# ---------------------------------------------------------------------------


def _richness_single(raw: np.ndarray, measures: list[str]) -> dict[str, float]:
    """Compute richness measures for one sample's count vector."""
    counts = raw.astype(float)
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
    from scipy.optimize import brentq

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
