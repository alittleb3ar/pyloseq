"""Tests for kOverA, filter_taxa, taxa_filter_mask, transform_sample_counts, rarefy_even_depth."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skbio
import skbio.tree

import pyloseq
from pyloseq import (
    OtuTable,
    Phyloseq,
    PhyTree,
    RefSeq,
    SampleData,
    TaxTable,
    filter_taxa,
    kOverA,
    rarefy_even_depth,
    taxa_filter_mask,
    transform_sample_counts,
)
from pyloseq.datasets.fixtures import load_enterotype_reference

GOLDEN_DIR = Path("tests/golden")
ET_GOLDEN = GOLDEN_DIR / "enterotype"
ET_FILTER_PRESENT = (
    ET_GOLDEN / "filter_taxa_kOverA_5_2e-5" / "otu_table.parquet"
).exists()


def _make_ps(
    ntaxa: int = 6,
    nsamples: int = 4,
    with_sam: bool = True,
    with_tax: bool = True,
    rng: np.random.Generator | None = None,
) -> Phyloseq:
    if rng is None:
        rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(ntaxa, nsamples)).astype(float)
    taxa = [f"OTU{i + 1}" for i in range(ntaxa)]
    samples = [f"S{i + 1}" for i in range(nsamples)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    otu = OtuTable(df, taxa_are_rows=True)

    sam = None
    if with_sam:
        sam_df = pd.DataFrame(
            {
                "Group": ["A", "A", "B", "B"][:nsamples],
                "Depth": [100.0, 200.0, 150.0, 250.0][:nsamples],
            },
            index=samples,
        )
        sam = SampleData(sam_df)

    tax = None
    if with_tax:
        phylum_vals = [
            "Firmicutes",
            "Firmicutes",
            "Bacteroidetes",
            "Proteobacteria",
            "Proteobacteria",
            "Chlamydiae",
        ][:ntaxa]
        genus_vals = ["Genus_A", "Genus_A", "Genus_B", "Genus_C", "Genus_D", "Genus_E"][
            :ntaxa
        ]
        tax_df = pd.DataFrame(
            {"Phylum": phylum_vals, "Genus": genus_vals},
            index=taxa,
        )
        tax = TaxTable(tax_df)

    return Phyloseq(otu=otu, sam=sam, tax=tax)


def _make_ps_with_tree() -> Phyloseq:
    newick = "((OTU1:0.1,OTU2:0.1):0.05,(OTU3:0.2,OTU4:0.1):0.05);"
    tree_node = skbio.tree.TreeNode.read(
        StringIO(newick), format="newick", convert_underscores=False
    )
    otu = OtuTable(
        pd.DataFrame(
            [[10, 5], [0, 20], [8, 3], [15, 1]],
            index=["OTU1", "OTU2", "OTU3", "OTU4"],
            columns=["S1", "S2"],
            dtype=float,
        ),
        taxa_are_rows=True,
    )
    return Phyloseq(otu=otu, tree=PhyTree(tree_node))


def _make_ps_with_refseq() -> Phyloseq:
    df = pd.DataFrame(
        {"S1": [10.0, 5.0, 0.0], "S2": [3.0, 8.0, 2.0]},
        index=["OTU1", "OTU2", "OTU3"],
    )
    rs = RefSeq(
        {
            "OTU1": skbio.DNA("ACGT"),
            "OTU2": skbio.DNA("TTTT"),
            "OTU3": skbio.DNA("GCGC"),
        }
    )
    return Phyloseq(otu=OtuTable(df, taxa_are_rows=True), refseq=rs)


@pytest.fixture
def ps() -> Phyloseq:
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


def test_filter_taxa_prune_false_returns_series(ps: Phyloseq) -> None:
    mask = taxa_filter_mask(ps, kOverA(1, 0.0))
    assert isinstance(mask, pd.Series)
    assert mask.index.equals(ps.taxa_names)


def test_filter_taxa_all_pass(ps: Phyloseq) -> None:
    ps2 = filter_taxa(ps, lambda x: True)
    assert ps2.ntaxa == ps.ntaxa


def test_filter_taxa_none_pass(ps: Phyloseq) -> None:
    ps2 = filter_taxa(ps, lambda x: False)
    assert ps2.ntaxa == 0


def test_filter_taxa_always_returns_phyloseq(ps: Phyloseq) -> None:
    result = filter_taxa(ps, kOverA(1, 5))
    assert isinstance(result, Phyloseq)


def test_taxa_filter_mask_returns_series(ps: Phyloseq) -> None:
    mask = taxa_filter_mask(ps, kOverA(1, 5))
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_filter_taxa_and_mask_consistent(ps: Phyloseq) -> None:
    pred = kOverA(1, 5)
    filtered = filter_taxa(ps, pred)
    mask = taxa_filter_mask(ps, pred)
    assert set(filtered.taxa_names) == set(mask.index[mask])


def test_taxa_filter_mask_exported() -> None:
    assert hasattr(pyloseq, "taxa_filter_mask")


@pytest.mark.skipif(not ET_FILTER_PRESENT, reason="golden files not generated yet")
def test_filter_taxa_kOverA_matches_r_enterotype() -> None:
    ref = load_enterotype_reference()
    et = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
    )
    et2 = filter_taxa(et, kOverA(5, 2e-5))

    golden_otu = pd.read_parquet(
        ET_GOLDEN / "filter_taxa_kOverA_5_2e-5" / "otu_table.parquet"
    )
    if "__index__" in golden_otu.columns:
        golden_otu = golden_otu.set_index("__index__")
        golden_otu.index.name = None
    assert et2.ntaxa == len(golden_otu)
    assert et2.nsamples == 280


# ===========================================================================
# transform_sample_counts
# ===========================================================================


def test_relative_abundance(ps: Phyloseq) -> None:
    ps2 = transform_sample_counts(ps, lambda x: x / x.sum())
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


def test_log_transform(ps: Phyloseq) -> None:
    ps2 = transform_sample_counts(ps, lambda x: np.log1p(x))
    orig = ps.otu_table.to_dataframe()
    result = ps2.otu_table.to_dataframe()
    np.testing.assert_allclose(result.values, np.log1p(orig.values), atol=1e-14)


def test_transform_no_mutation(ps: Phyloseq) -> None:
    orig_sum = ps.taxa_sums().sum()
    _ = transform_sample_counts(ps, lambda x: x / x.sum())
    assert ps.taxa_sums().sum() == orig_sum


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
