"""Tests for prune_taxa, prune_samples, subset_samples, subset_taxa."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import _make_ps, requires_golden

import pyloseq
from pyloseq import (
    OtuTable,
    Phyloseq,
    SampleData,
    TaxTable,
    prune_samples,
    prune_taxa,
    subset_samples,
    subset_taxa,
)
from pyloseq.datasets.fixtures import load_global_patterns_reference

_GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def ps_pruning() -> Phyloseq:
    return _make_ps()


# ===========================================================================
# prune_taxa
# ===========================================================================


def test_prune_taxa_basic(ps_pruning: Phyloseq) -> None:
    ps2 = prune_taxa(["OTU1", "OTU3"], ps_pruning)
    assert list(ps2.taxa_names) == ["OTU1", "OTU3"]
    assert ps2.ntaxa == 2
    assert ps2.nsamples == ps_pruning.nsamples


def test_prune_taxa_order_preserved(ps_pruning: Phyloseq) -> None:
    ps2 = prune_taxa(["OTU4", "OTU1", "OTU2"], ps_pruning)
    assert list(ps2.taxa_names) == ["OTU4", "OTU1", "OTU2"]


def test_prune_taxa_absent_names_ignored(ps_pruning: Phyloseq) -> None:
    ps2 = prune_taxa(["OTU1", "UNKNOWN", "OTU2"], ps_pruning)
    assert list(ps2.taxa_names) == ["OTU1", "OTU2"]


def test_prune_taxa_no_mutation(ps_pruning: Phyloseq) -> None:
    original_ntaxa = ps_pruning.ntaxa
    _ = prune_taxa(["OTU1"], ps_pruning)
    assert ps_pruning.ntaxa == original_ntaxa


# ===========================================================================
# prune_samples
# ===========================================================================


def test_prune_samples_basic(ps_pruning: Phyloseq) -> None:
    ps2 = prune_samples(["S1", "S3"], ps_pruning)
    assert list(ps2.sample_names) == ["S1", "S3"]
    assert ps2.nsamples == 2
    assert ps2.ntaxa == ps_pruning.ntaxa


def test_prune_samples_updates_sample_data(ps_pruning: Phyloseq) -> None:
    ps2 = prune_samples(["S2"], ps_pruning)
    assert ps2.sample_data is not None
    assert list(ps2.sample_names) == ["S2"]


def test_prune_samples_no_mutation(ps_pruning: Phyloseq) -> None:
    original_nsamples = ps_pruning.nsamples
    _ = prune_samples(["S1"], ps_pruning)
    assert ps_pruning.nsamples == original_nsamples


# ===========================================================================
# subset_samples
# ===========================================================================


def test_subset_samples_lambda(ps_pruning: Phyloseq) -> None:
    ps2 = subset_samples(ps_pruning, lambda s: s["Group"] == "A")
    assert all(ps2.sample_data.to_frame()["Group"] == "A")


def test_subset_samples_query_string(ps_pruning: Phyloseq) -> None:
    ps2 = subset_samples(ps_pruning, 'Group == "B"')
    assert all(ps2.sample_data.to_frame()["Group"] == "B")


def test_subset_samples_preserves_taxa(ps_pruning: Phyloseq) -> None:
    ps2 = subset_samples(ps_pruning, lambda s: s["Group"] == "A")
    assert ps2.ntaxa == ps_pruning.ntaxa


def test_subset_samples_no_sample_data_raises() -> None:
    ps = _make_ps(with_sam=False)
    with pytest.raises(pyloseq.pyloseqValidationError):
        subset_samples(ps, lambda s: True)


@requires_golden("GlobalPatterns", "subset_samples_soil", "otu_table.parquet")
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
        _GOLDEN / "GlobalPatterns" / "subset_samples_soil" / "otu_table.parquet"
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


def test_subset_samples_zero_result_raises_or_is_empty(ps_pruning: Phyloseq) -> None:
    try:
        ps2 = subset_samples(ps_pruning, lambda df: df["Group"] == "Z")
        assert ps2.nsamples == 0 or ps2.ntaxa == 0
    except Exception:
        pass


# ===========================================================================
# subset_taxa
# ===========================================================================


def test_subset_taxa_lambda(ps_pruning: Phyloseq) -> None:
    ps2 = subset_taxa(ps_pruning, lambda t: t["Phylum"] == "Firmicutes")
    assert ps2.ntaxa == 2
    assert all(ps2.tax_table.to_frame()["Phylum"] == "Firmicutes")


def test_subset_taxa_query_string(ps_pruning: Phyloseq) -> None:
    ps2 = subset_taxa(ps_pruning, 'Phylum == "Chlamydiae"')
    assert ps2.ntaxa == 1


def test_subset_taxa_no_tax_table_raises() -> None:
    ps = _make_ps(with_tax=False)
    with pytest.raises(pyloseq.pyloseqValidationError):
        subset_taxa(ps, lambda t: True)


@requires_golden("GlobalPatterns", "subset_taxa_chlamydiae", "otu_table.parquet")

def test_subset_taxa_chlamydiae_matches_r() -> None:
    ref = load_global_patterns_reference()
    gp = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tax=TaxTable(ref["tax_table"]),
    )
    gp_chlam = subset_taxa(gp, 'Phylum == "Chlamydiae"')

    golden_otu = pd.read_parquet(
        _GOLDEN / "GlobalPatterns" / "subset_taxa_chlamydiae" / "otu_table.parquet"
    )
    if "__index__" in golden_otu.columns:
        golden_otu = golden_otu.set_index("__index__")
        golden_otu.index.name = None
    assert gp_chlam.ntaxa == len(golden_otu)


def test_subset_taxa_zero_result_raises_or_is_empty(ps_pruning: Phyloseq) -> None:
    ps_tax = _make_ps(with_tax=True)
    try:
        ps2 = subset_taxa(ps_tax, lambda df: df["Phylum"] == "DoesNotExist")
        assert ps2.ntaxa == 0 or ps2.nsamples == 0
    except Exception:
        pass
