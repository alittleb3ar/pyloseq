"""Tests for kOverA, filter_taxa, taxa_filter_mask, transform_sample_counts, rarefy_even_depth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import _make_ps, _make_ps_with_refseq, _make_ps_with_tree, requires_golden

import pyloseq
from pyloseq import (
    OtuTable,
    Phyloseq,
    SampleData,
    filter_taxa,
    kOverA,
    rarefy_even_depth,
    taxa_filter_mask,
    transform_sample_counts,
)
from pyloseq.datasets.fixtures import load_enterotype_reference

_GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def ps_transform() -> Phyloseq:
    return _make_ps()


# ===========================================================================
# kOverA predicate
# ===========================================================================


def test_kOverA_basic() -> None:
    pred = kOverA(2, 10.0)
    assert pred(pd.Series([0.0, 11.0, 12.0, 5.0])) is True
    assert pred(pd.Series([0.0, 11.0, 5.0, 5.0])) is False


# ===========================================================================
# filter_taxa / taxa_filter_mask
# ===========================================================================


def test_filter_taxa_prune_true() -> None:
    ps = _make_ps(rng=np.random.default_rng(1))
    ps2 = filter_taxa(ps, kOverA(1, 0.0))
    assert isinstance(ps2, Phyloseq)
    assert ps2.ntaxa <= ps.ntaxa


def test_filter_taxa_prune_false_returns_series(ps_transform: Phyloseq) -> None:
    mask = taxa_filter_mask(ps_transform, kOverA(1, 0.0))
    assert isinstance(mask, pd.Series)
    assert mask.index.equals(ps_transform.taxa_names)


def test_filter_taxa_all_pass(ps_transform: Phyloseq) -> None:
    ps2 = filter_taxa(ps_transform, lambda x: True)
    assert ps2.ntaxa == ps_transform.ntaxa


def test_filter_taxa_none_pass(ps_transform: Phyloseq) -> None:
    ps2 = filter_taxa(ps_transform, lambda x: False)
    assert ps2.ntaxa == 0


def test_filter_taxa_always_returns_phyloseq(ps_transform: Phyloseq) -> None:
    result = filter_taxa(ps_transform, kOverA(1, 5))
    assert isinstance(result, Phyloseq)


def test_taxa_filter_mask_returns_series(ps_transform: Phyloseq) -> None:
    mask = taxa_filter_mask(ps_transform, kOverA(1, 5))
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_filter_taxa_and_mask_consistent(ps_transform: Phyloseq) -> None:
    pred = kOverA(1, 5)
    filtered = filter_taxa(ps_transform, pred)
    mask = taxa_filter_mask(ps_transform, pred)
    assert set(filtered.taxa_names) == set(mask.index[mask])


def test_taxa_filter_mask_exported() -> None:
    assert hasattr(pyloseq, "taxa_filter_mask")


@requires_golden("enterotype", "filter_taxa_kOverA_5_2e-5", "otu_table.parquet")
def test_filter_taxa_kOverA_matches_r_enterotype() -> None:
    ref = load_enterotype_reference()
    et = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
    )
    et2 = filter_taxa(et, kOverA(5, 2e-5))

    golden_otu = pd.read_parquet(
        _GOLDEN / "enterotype" / "filter_taxa_kOverA_5_2e-5" / "otu_table.parquet"
    )
    if "__index__" in golden_otu.columns:
        golden_otu = golden_otu.set_index("__index__")
        golden_otu.index.name = None
    assert et2.ntaxa == len(golden_otu)
    assert et2.nsamples == 280


# ===========================================================================
# transform_sample_counts
# ===========================================================================


def test_relative_abundance(ps_transform: Phyloseq) -> None:
    ps2 = transform_sample_counts(ps_transform, lambda x: x / x.sum())
    col_sums = ps2.otu_table.to_dataframe().sum(axis=0)
    np.testing.assert_allclose(col_sums.values, 1.0, atol=1e-12)


def test_all_zero_sample_produces_nan() -> None:
    df = pd.DataFrame(
        {"S1": [0.0, 0.0], "S2": [5.0, 3.0]},
        index=["OTU1", "OTU2"],
    )
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    ps2 = transform_sample_counts(ps, lambda x: x / x.sum())
    assert ps2.otu_table.to_dataframe()["S1"].isna().all()


def test_log_transform(ps_transform: Phyloseq) -> None:
    ps2 = transform_sample_counts(ps_transform, lambda x: np.log1p(x))
    orig = ps_transform.otu_table.to_dataframe()
    result = ps2.otu_table.to_dataframe()
    np.testing.assert_allclose(result.values, np.log1p(orig.values), atol=1e-14)


def test_transform_no_mutation(ps_transform: Phyloseq) -> None:
    orig_sum = ps_transform.taxa_sums().sum()
    _ = transform_sample_counts(ps_transform, lambda x: x / x.sum())
    assert ps_transform.taxa_sums().sum() == orig_sum


# ===========================================================================
# rarefy_even_depth
# ===========================================================================


def test_rarefy_all_samples_equal_depth() -> None:
    ps = _make_ps(rng=np.random.default_rng(99))
    depth = 20
    ps2 = rarefy_even_depth(ps, sample_size=depth, verbose=False)
    ss = ps2.sample_sums()
    np.testing.assert_allclose(ss.values, depth, atol=0)


def test_rarefy_low_samples_dropped() -> None:
    df = pd.DataFrame(
        {"S1": [5.0, 3.0], "S2": [100.0, 200.0]},
        index=["OTU1", "OTU2"],
    )
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    with pytest.warns(UserWarning, match="Dropping"):
        ps2 = rarefy_even_depth(ps, sample_size=50, verbose=True)
    assert "S1" not in list(ps2.sample_names)
    assert "S2" in list(ps2.sample_names)


def test_rarefy_reproducibility() -> None:
    ps = _make_ps(rng=np.random.default_rng(7))
    ps_a = rarefy_even_depth(ps, sample_size=15, rng_seed=42, verbose=False)
    ps_b = rarefy_even_depth(ps, sample_size=15, rng_seed=42, verbose=False)
    np.testing.assert_array_equal(
        ps_a.otu_table.to_dataframe().values,
        ps_b.otu_table.to_dataframe().values,
    )


def test_rarefy_trim_otus() -> None:
    df = pd.DataFrame(
        {"S1": [100.0, 0.0, 50.0], "S2": [200.0, 0.0, 30.0]},
        index=["OTU1", "OTU2", "OTU3"],
    )
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    ps2 = rarefy_even_depth(ps, sample_size=20, trim_otus=True, verbose=False)
    assert "OTU2" not in list(ps2.taxa_names)


def test_rarefy_replace_false() -> None:
    df = pd.DataFrame(
        {"S1": [50.0, 30.0, 20.0], "S2": [60.0, 20.0, 20.0]},
        index=["OTU1", "OTU2", "OTU3"],
    )
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    ps2 = rarefy_even_depth(ps, sample_size=10, replace=False, verbose=False)
    ss = ps2.sample_sums()
    np.testing.assert_allclose(ss.values, 10.0, atol=0)


def test_rarefy_no_mutation() -> None:
    ps = _make_ps(rng=np.random.default_rng(3))
    orig = ps.taxa_sums().sum()
    _ = rarefy_even_depth(ps, sample_size=10, verbose=False)
    assert ps.taxa_sums().sum() == orig


def test_rarefy_preserves_refseq() -> None:
    ps = _make_ps_with_refseq()
    ps2 = rarefy_even_depth(ps, sample_size=5, rng_seed=1, verbose=False)
    assert ps2.refseq is not None


def test_rarefy_preserves_tree() -> None:
    ps = _make_ps_with_tree()
    ps2 = rarefy_even_depth(ps, sample_size=5, rng_seed=1, verbose=False)
    assert ps2.phy_tree is not None


# ===========================================================================
# API surface
# ===========================================================================


def test_manipulation_functions_exported() -> None:
    for name in [
        "subset_samples",
        "subset_taxa",
        "prune_samples",
        "prune_taxa",
        "filter_taxa",
        "kOverA",
        "transform_sample_counts",
        "rarefy_even_depth",
        "tax_glom",
        "tip_glom",
        "merge_phyloseq",
        "merge_samples",
        "merge_taxa",
        "psmelt",
    ]:
        assert hasattr(pyloseq, name), f"pyloseq.{name} not exported"
