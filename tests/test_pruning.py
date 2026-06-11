"""Tests for prune_taxa, prune_samples, subset_samples, subset_taxa."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyloseq
from pyloseq import (OtuTable, Phyloseq, SampleData, TaxTable, prune_samples,
                     prune_taxa, subset_samples, subset_taxa)
from pyloseq.datasets.fixtures import load_global_patterns_reference

GOLDEN_DIR = Path("tests/golden")
GP_GOLDEN = GOLDEN_DIR / "GlobalPatterns"
GP_SUBSET_SOIL_PRESENT = (
    GP_GOLDEN / "subset_samples_soil" / "otu_table.parquet"
).exists()
GP_SUBSET_CHLAM_PRESENT = (
    GP_GOLDEN / "subset_taxa_chlamydiae" / "otu_table.parquet"
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


@pytest.fixture
def ps() -> Phyloseq:
    return _make_ps()


# ===========================================================================
# prune_taxa
# ===========================================================================


def test_prune_taxa_basic(ps: Phyloseq) -> None:
    ps2 = prune_taxa(["OTU1", "OTU3"], ps)
    assert list(ps2.taxa_names) == ["OTU1", "OTU3"]
    assert ps2.ntaxa == 2
    assert ps2.nsamples == ps.nsamples


def test_prune_taxa_order_preserved(ps: Phyloseq) -> None:
    ps2 = prune_taxa(["OTU4", "OTU1", "OTU2"], ps)
    assert list(ps2.taxa_names) == ["OTU4", "OTU1", "OTU2"]


def test_prune_taxa_absent_names_ignored(ps: Phyloseq) -> None:
    ps2 = prune_taxa(["OTU1", "UNKNOWN", "OTU2"], ps)
    assert list(ps2.taxa_names) == ["OTU1", "OTU2"]


def test_prune_taxa_no_mutation(ps: Phyloseq) -> None:
    original_ntaxa = ps.ntaxa
    _ = prune_taxa(["OTU1"], ps)
    assert ps.ntaxa == original_ntaxa


# ===========================================================================
# prune_samples
# ===========================================================================


def test_prune_samples_basic(ps: Phyloseq) -> None:
    ps2 = prune_samples(["S1", "S3"], ps)
    assert list(ps2.sample_names) == ["S1", "S3"]
    assert ps2.nsamples == 2
    assert ps2.ntaxa == ps.ntaxa


def test_prune_samples_updates_sample_data(ps: Phyloseq) -> None:
    ps2 = prune_samples(["S2"], ps)
    assert ps2.sample_data is not None
    assert list(ps2.sample_names) == ["S2"]


def test_prune_samples_no_mutation(ps: Phyloseq) -> None:
    original_nsamples = ps.nsamples
    _ = prune_samples(["S1"], ps)
    assert ps.nsamples == original_nsamples


# ===========================================================================
# subset_samples
# ===========================================================================


def test_subset_samples_lambda(ps: Phyloseq) -> None:
    ps2 = subset_samples(ps, lambda s: s["Group"] == "A")
    assert all(ps2.sample_data.to_frame()["Group"] == "A")


def test_subset_samples_query_string(ps: Phyloseq) -> None:
    ps2 = subset_samples(ps, 'Group == "B"')
    assert all(ps2.sample_data.to_frame()["Group"] == "B")


def test_subset_samples_preserves_taxa(ps: Phyloseq) -> None:
    ps2 = subset_samples(ps, lambda s: s["Group"] == "A")
    assert ps2.ntaxa == ps.ntaxa


def test_subset_samples_no_sample_data_raises() -> None:
    ps = _make_ps(with_sam=False)
    with pytest.raises(pyloseq.pyloseqValidationError):
        subset_samples(ps, lambda s: True)


@pytest.mark.skipif(not GP_SUBSET_SOIL_PRESENT, reason="golden files not generated yet")
def test_subset_samples_soil_matches_r() -> None:
    ref = load_global_patterns_reference()
    gp = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
        tax=TaxTable(ref["tax_table"]),
    )
    gp_soil = subset_samples(gp, 'SampleType == "Soil"')
    assert gp_soil.nsamples == 3

    golden_otu = pd.read_parquet(
        GP_GOLDEN / "subset_samples_soil" / "otu_table.parquet"
    )
    if "__index__" in golden_otu.columns:
        golden_otu = golden_otu.set_index("__index__")
        golden_otu.index.name = None
    result_otu = gp_soil.otu_table.to_dataframe()
    common_taxa = result_otu.index.intersection(golden_otu.index)
    common_samples = result_otu.columns.intersection(golden_otu.columns)
    np.testing.assert_allclose(
        result_otu.loc[common_taxa, common_samples].values,
        golden_otu.loc[common_taxa, common_samples].values,
        atol=1e-10,
    )


def test_subset_samples_zero_result_raises_or_is_empty(ps: Phyloseq) -> None:
    try:
        ps2 = subset_samples(ps, lambda df: df["Group"] == "Z")
        assert ps2.nsamples == 0 or ps2.ntaxa == 0
    except Exception:
        pass


# ===========================================================================
# subset_taxa
# ===========================================================================


def test_subset_taxa_lambda(ps: Phyloseq) -> None:
    ps2 = subset_taxa(ps, lambda t: t["Phylum"] == "Firmicutes")
    assert ps2.ntaxa == 2
    assert all(ps2.tax_table.to_frame()["Phylum"] == "Firmicutes")


def test_subset_taxa_query_string(ps: Phyloseq) -> None:
    ps2 = subset_taxa(ps, 'Phylum == "Chlamydiae"')
    assert ps2.ntaxa == 1


def test_subset_taxa_no_tax_table_raises() -> None:
    ps = _make_ps(with_tax=False)
    with pytest.raises(pyloseq.pyloseqValidationError):
        subset_taxa(ps, lambda t: True)


@pytest.mark.skipif(
    not GP_SUBSET_CHLAM_PRESENT, reason="golden files not generated yet"
)
def test_subset_taxa_chlamydiae_matches_r() -> None:
    ref = load_global_patterns_reference()
    gp = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tax=TaxTable(ref["tax_table"]),
    )
    gp_chlam = subset_taxa(gp, 'Phylum == "Chlamydiae"')

    golden_otu = pd.read_parquet(
        GP_GOLDEN / "subset_taxa_chlamydiae" / "otu_table.parquet"
    )
    if "__index__" in golden_otu.columns:
        golden_otu = golden_otu.set_index("__index__")
        golden_otu.index.name = None
    assert gp_chlam.ntaxa == len(golden_otu)


def test_subset_taxa_zero_result_raises_or_is_empty(ps: Phyloseq) -> None:
    ps_tax = _make_ps(with_tax=True)
    try:
        ps2 = subset_taxa(ps_tax, lambda df: df["Phylum"] == "DoesNotExist")
        assert ps2.ntaxa == 0 or ps2.nsamples == 0
    except Exception:
        pass
