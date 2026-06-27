"""Tests for beta diversity distances: dispatcher, UniFrac, JSD, DPCoA, kind parameter."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import requires_golden
from skbio.stats.distance import DistanceMatrix

import pyloseq
from pyloseq import OtuTable, Phyloseq, PhyTree, distance, distance_method_list, unifrac
from pyloseq.datasets.fixtures import load_esophagus_reference

_GOLDEN = Path(__file__).parent / "golden"


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
# distance dispatcher — catalogue and error
# ===========================================================================


def test_distance_method_list_returns_dict() -> None:
    ml = distance_method_list()
    assert isinstance(ml, dict)
    assert "vegan-equivalent" in ml
    assert "phylogenetic" in ml


def test_unknown_method_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        distance(ps, "not_a_method")


# ===========================================================================
# scipy metrics — type, dimensions, symmetry, zero diagonal
# ===========================================================================


@pytest.mark.parametrize(
    "method",
    ["bray", "euclidean", "jaccard", "jsd", "cosine", "canberra"],
)
def test_metric_properties(ps: Phyloseq, method: str) -> None:
    """Each metric returns a symmetric DistanceMatrix with a zero diagonal."""
    dm = distance(ps, method)
    arr = np.array(dm.data)
    assert isinstance(dm, DistanceMatrix)
    assert len(dm.ids) == ps.nsamples
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)
    np.testing.assert_allclose(arr, arr.T, atol=1e-10)


def test_jaccard_values_in_unit_range(ps: Phyloseq) -> None:
    """Jaccard (presence/absence) distances are bounded to [0, 1]."""
    dm = distance(ps, "jaccard")
    arr = np.array(dm.data)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0 + 1e-12


# ===========================================================================
# UniFrac — type, dimensions, symmetry, zero diagonal
# ===========================================================================


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_unifrac_properties(ps_with_tree: Phyloseq, weighted: bool) -> None:
    """UniFrac returns a symmetric DistanceMatrix with a zero diagonal."""
    dm = unifrac(ps_with_tree, weighted=weighted)
    arr = np.array(dm.data)
    assert isinstance(dm, DistanceMatrix)
    assert len(dm.ids) == ps_with_tree.nsamples
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)
    np.testing.assert_allclose(arr, arr.T, atol=1e-12)


def test_unifrac_requires_tree(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        unifrac(ps)


def test_unifrac_via_distance(ps_with_tree: Phyloseq) -> None:
    dm1 = unifrac(ps_with_tree, weighted=False)
    dm2 = distance(ps_with_tree, "unifrac")
    np.testing.assert_allclose(np.array(dm1.data), np.array(dm2.data), atol=1e-12)


@requires_golden("esophagus", "unifrac_unweighted", "normalized.parquet")

def test_unweighted_unifrac_matches_r_esophagus() -> None:
    ps = _load_esophagus()
    dm = unifrac(ps, weighted=False, normalized=True)

    golden = pd.read_parquet(_GOLDEN / "esophagus" / "unifrac_unweighted" / "normalized.parquet")
    if "__index__" in golden.columns:
        golden = golden.set_index("__index__")
        golden.index.name = None

    ids = list(dm.ids)
    for u in ids:
        for v in ids:
            if u in golden.index and v in golden.columns:
                np.testing.assert_allclose(
                    dm[u, v],
                    float(golden.loc[u, v]),
                    atol=1e-4,
                    err_msg=f"UniFrac mismatch: {u} vs {v}",
                )


@requires_golden("esophagus", "unifrac_weighted", "normalized.parquet")

def test_weighted_unifrac_matches_r_esophagus() -> None:
    ps = _load_esophagus()
    dm = unifrac(ps, weighted=True, normalized=True)

    golden = pd.read_parquet(_GOLDEN / "esophagus" / "unifrac_weighted" / "normalized.parquet")
    if "__index__" in golden.columns:
        golden = golden.set_index("__index__")
        golden.index.name = None

    ids = list(dm.ids)
    for u in ids:
        for v in ids:
            if u in golden.index and v in golden.columns:
                np.testing.assert_allclose(
                    dm[u, v],
                    float(golden.loc[u, v]),
                    atol=1e-4,
                    err_msg=f"Weighted UniFrac mismatch: {u} vs {v}",
                )


# ===========================================================================
# JSD — domain-specific properties
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
# DPCoA
# ===========================================================================


def test_dpcoa_returns_dm(ps_with_tree: Phyloseq) -> None:
    dm = distance(ps_with_tree, "dpcoa")
    assert isinstance(dm, DistanceMatrix)
    arr = np.array(dm.data)
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)
    np.testing.assert_allclose(arr, arr.T, atol=1e-10)


# ===========================================================================
# kind parameter — samples vs taxa
# ===========================================================================


def test_distance_type_deprecated(ps: Phyloseq) -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        distance(ps, "bray", type="samples")
        assert any("deprecated" in str(x.message).lower() for x in w)


@pytest.mark.parametrize("method", ["bray", "euclidean"])
def test_taxa_kind_properties(ps: Phyloseq, method: str) -> None:
    """kind='taxa' produces an ntaxa × ntaxa symmetric distance matrix."""
    dm = distance(ps, method, kind="taxa")
    arr = np.array(dm.data)
    assert isinstance(dm, DistanceMatrix)
    assert len(dm.ids) == ps.ntaxa
    assert set(dm.ids) == set(ps.taxa_names)
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)
    np.testing.assert_allclose(arr, arr.T, atol=1e-10)
