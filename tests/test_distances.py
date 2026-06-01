"""Tests for beta diversity distances: dispatcher, UniFrac, JSD, DPCoA, kind parameter."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skbio.stats.distance import DistanceMatrix

import pyloseq
from pyloseq import (OtuTable, Phyloseq, PhyTree, distance,
                     distance_method_list, unifrac)
from pyloseq.datasets.fixtures import load_esophagus_reference

GOLDEN_DIR = Path("tests/golden")
ES_GOLDEN = GOLDEN_DIR / "esophagus"
ES_PRESENT = (ES_GOLDEN / "otu_table.parquet").exists()
UF_UN_PRESENT = (ES_GOLDEN / "unifrac_unweighted" / "normalized.parquet").exists()
UF_WT_PRESENT = (ES_GOLDEN / "unifrac_weighted" / "normalized.parquet").exists()


def _load_esophagus() -> Phyloseq:

    ref = load_esophagus_reference()
    tree = (
        PhyTree.from_newick(ref["phy_tree_newick"])
        if "phy_tree_newick" in ref
        else None
    )
    return Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tree=tree,
    )


# ===========================================================================
# distance dispatcher
# ===========================================================================


def test_distance_method_list_returns_dict() -> None:
    ml = distance_method_list()
    assert isinstance(ml, dict)
    assert "vegan-equivalent" in ml
    assert "phylogenetic" in ml


def test_bray_returns_dm(ps: Phyloseq) -> None:

    dm = distance(ps, "bray")
    assert isinstance(dm, DistanceMatrix)
    assert len(dm.ids) == ps.nsamples


def test_euclidean(ps: Phyloseq) -> None:

    dm = distance(ps, "euclidean")
    assert isinstance(dm, DistanceMatrix)


def test_jaccard_binary(ps: Phyloseq) -> None:
    dm = distance(ps, "jaccard")
    arr = np.array(dm.data)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0 + 1e-12


def test_jsd(ps: Phyloseq) -> None:

    dm = distance(ps, "jsd")
    assert isinstance(dm, DistanceMatrix)


def test_unknown_method_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        distance(ps, "not_a_method")


def test_symmetric(ps: Phyloseq) -> None:
    dm = distance(ps, "bray")
    arr = np.array(dm.data)
    np.testing.assert_allclose(arr, arr.T, atol=1e-12)


def test_zero_diagonal(ps: Phyloseq) -> None:
    dm = distance(ps, "bray")
    arr = np.array(dm.data)
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)


# ===========================================================================
# UniFrac
# ===========================================================================


def test_unifrac_unweighted(ps_with_tree: Phyloseq) -> None:

    dm = unifrac(ps_with_tree, weighted=False)
    assert isinstance(dm, DistanceMatrix)
    assert len(dm.ids) == ps_with_tree.nsamples


def test_unifrac_weighted(ps_with_tree: Phyloseq) -> None:

    dm = unifrac(ps_with_tree, weighted=True)
    assert isinstance(dm, DistanceMatrix)


def test_unifrac_requires_tree(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        unifrac(ps)


def test_unifrac_symmetric(ps_with_tree: Phyloseq) -> None:
    dm = unifrac(ps_with_tree, weighted=False)
    arr = np.array(dm.data)
    np.testing.assert_allclose(arr, arr.T, atol=1e-12)


def test_unifrac_zero_diagonal(ps_with_tree: Phyloseq) -> None:
    dm = unifrac(ps_with_tree, weighted=False)
    arr = np.array(dm.data)
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)


def test_unifrac_via_distance(ps_with_tree: Phyloseq) -> None:
    dm1 = unifrac(ps_with_tree, weighted=False)
    dm2 = distance(ps_with_tree, "unifrac")
    np.testing.assert_allclose(np.array(dm1.data), np.array(dm2.data), atol=1e-12)


@pytest.mark.skipif(
    not (ES_PRESENT and UF_UN_PRESENT), reason="golden files not generated yet"
)
def test_unweighted_unifrac_matches_r_esophagus() -> None:
    ps = _load_esophagus()
    dm = unifrac(ps, weighted=False, normalized=True)

    golden = pd.read_parquet(ES_GOLDEN / "unifrac_unweighted" / "normalized.parquet")
    if "__index__" in golden.columns:
        golden = golden.set_index("__index__")
        golden.index.name = None

    ids = list(dm.ids)
    for _i, u in enumerate(ids):
        for _j, v in enumerate(ids):
            if u in golden.index and v in golden.columns:
                np.testing.assert_allclose(
                    dm[u, v],
                    float(golden.loc[u, v]),
                    atol=1e-4,
                    err_msg=f"UniFrac mismatch: {u} vs {v}",
                )


@pytest.mark.skipif(
    not (ES_PRESENT and UF_WT_PRESENT), reason="golden files not generated yet"
)
def test_weighted_unifrac_matches_r_esophagus() -> None:
    ps = _load_esophagus()
    dm = unifrac(ps, weighted=True, normalized=True)

    golden = pd.read_parquet(ES_GOLDEN / "unifrac_weighted" / "normalized.parquet")
    if "__index__" in golden.columns:
        golden = golden.set_index("__index__")
        golden.index.name = None

    ids = list(dm.ids)
    for _i, u in enumerate(ids):
        for _j, v in enumerate(ids):
            if u in golden.index and v in golden.columns:
                np.testing.assert_allclose(
                    dm[u, v],
                    float(golden.loc[u, v]),
                    atol=1e-4,
                    err_msg=f"Weighted UniFrac mismatch: {u} vs {v}",
                )


# ===========================================================================
# JSD is a true distance in [0, 1]
# ===========================================================================


def test_jsd_in_unit_range(ps: Phyloseq) -> None:
    dm = distance(ps, "jsd")
    arr = np.array(dm.data)
    assert arr.min() >= 0.0 - 1e-12
    assert arr.max() <= 1.0 + 1e-12


def test_jsd_identical_samples_zero() -> None:
    df = pd.DataFrame(
        {"S1": [10.0, 20.0, 5.0], "S2": [10.0, 20.0, 5.0]}, index=["A", "B", "C"]
    )
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    dm = distance(ps, "jsd")
    np.testing.assert_allclose(dm["S1", "S2"], 0.0, atol=1e-12)


def test_jsd_completely_disjoint_is_one() -> None:
    df = pd.DataFrame({"S1": [10.0, 0.0], "S2": [0.0, 10.0]}, index=["A", "B"])
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    dm = distance(ps, "jsd")
    np.testing.assert_allclose(dm["S1", "S2"], 1.0, atol=1e-10)


# ===========================================================================
# DPCoA returns a valid DistanceMatrix
# ===========================================================================


def test_dpcoa_returns_dm(ps_with_tree: Phyloseq) -> None:

    dm = distance(ps_with_tree, "dpcoa")
    assert isinstance(dm, DistanceMatrix)
    arr = np.array(dm.data)
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)
    np.testing.assert_allclose(arr, arr.T, atol=1e-10)


# ===========================================================================
# 'kind' parameter and deprecated 'type' alias
# ===========================================================================


def test_distance_kind_parameter(ps: Phyloseq) -> None:
    dm = distance(ps, "bray", kind="samples")
    assert len(dm.ids) == ps.nsamples


def test_distance_type_deprecated(ps: Phyloseq) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        distance(ps, "bray", type="samples")
        assert any("deprecated" in str(x.message).lower() for x in w)
