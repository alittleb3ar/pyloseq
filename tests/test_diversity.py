"""Tests for alpha diversity estimation (estimate_richness, se.ACE)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyloseq
from pyloseq import OtuTable, Phyloseq, estimate_richness
from pyloseq.datasets.fixtures import load_global_patterns_reference

GOLDEN_DIR = Path("tests/golden")
GP_GOLDEN = GOLDEN_DIR / "GlobalPatterns"
RICH_PRESENT = (GP_GOLDEN / "estimate_richness" / "default.parquet").exists()


# ===========================================================================
# estimate_richness
# ===========================================================================


def test_estimate_richness_returns_dataframe(ps: Phyloseq) -> None:
    df = estimate_richness(ps)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == ps.nsamples


def test_estimate_richness_default_measures(ps: Phyloseq) -> None:
    df = estimate_richness(ps)
    expected = [
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
    assert list(df.columns) == expected


def test_estimate_richness_subset_measures(ps: Phyloseq) -> None:
    df = estimate_richness(ps, measures=["Observed", "Shannon"])
    assert list(df.columns) == ["Observed", "Shannon"]


def test_observed_is_nonzero_taxa() -> None:
    df_data = pd.DataFrame(
        {"S1": [10.0, 0.0, 5.0], "S2": [0.0, 3.0, 7.0]},
        index=["OTU1", "OTU2", "OTU3"],
    )
    ps = Phyloseq(otu=OtuTable(df_data, taxa_are_rows=True))
    df = estimate_richness(ps, measures=["Observed"])
    assert df.loc["S1", "Observed"] == 2.0
    assert df.loc["S2", "Observed"] == 2.0


def test_shannon_relative_entropy() -> None:
    df_data = pd.DataFrame(
        {"S1": [25.0, 25.0, 25.0, 25.0]},
        index=["OTU1", "OTU2", "OTU3", "OTU4"],
    )
    ps = Phyloseq(otu=OtuTable(df_data, taxa_are_rows=True))
    df = estimate_richness(ps, measures=["Shannon"])
    np.testing.assert_allclose(df.loc["S1", "Shannon"], np.log(4), atol=1e-12)


def test_simpson_range(ps: Phyloseq) -> None:
    df = estimate_richness(ps, measures=["Simpson"])
    assert ((df["Simpson"] >= 0) & (df["Simpson"] <= 1)).all()


def test_invsimpson_reciprocal(ps: Phyloseq) -> None:
    df = estimate_richness(ps, measures=["Simpson", "InvSimpson"])
    d = 1.0 - df["Simpson"].values
    expected_inv = 1.0 / d
    np.testing.assert_allclose(df["InvSimpson"].values, expected_inv, atol=1e-12)


def test_fisher_positive(ps: Phyloseq) -> None:
    df = estimate_richness(ps, measures=["Fisher"])
    assert (df["Fisher"] > 0).all()


def test_chao1_ge_observed(ps: Phyloseq) -> None:
    df = estimate_richness(ps, measures=["Observed", "Chao1"])
    assert (df["Chao1"] >= df["Observed"]).all()


def test_bad_measure_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        estimate_richness(ps, measures=["NotAMeasure"])


@pytest.mark.skipif(not RICH_PRESENT, reason="golden files not generated yet")
def test_estimate_richness_matches_r_globalpatterns() -> None:

    ref = load_global_patterns_reference()
    gp = Phyloseq(otu=OtuTable(ref["otu_table"], taxa_are_rows=True))
    result = estimate_richness(gp)

    golden = pd.read_parquet(GP_GOLDEN / "estimate_richness" / "default.parquet")
    if "__index__" in golden.columns:
        golden = golden.set_index("__index__")
        golden.index.name = None

    common_samples = result.index.intersection(golden.index)
    for measure in ["Observed", "Shannon", "Simpson", "InvSimpson"]:
        if measure in golden.columns:
            np.testing.assert_allclose(
                result.loc[common_samples, measure].values,
                golden.loc[common_samples, measure].values,
                atol=1e-9,
                err_msg=f"Mismatch for measure: {measure}",
            )

    if "Fisher" in golden.columns:
        np.testing.assert_allclose(
            result.loc[common_samples, "Fisher"].values,
            golden.loc[common_samples, "Fisher"].values,
            atol=1e-4,
        )


# ===========================================================================
# se.ACE warns; ACE is not NaN
# ===========================================================================


def test_se_ace_warns(ps: Phyloseq) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimate_richness(ps, measures=["se.ACE"])
        assert any("se.ACE" in str(x.message) for x in w)


def test_ace_is_not_all_nan(ps: Phyloseq) -> None:
    df = estimate_richness(ps, measures=["ACE"])
    assert not df["ACE"].isna().all()
