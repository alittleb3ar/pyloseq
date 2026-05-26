"""Tests for beta diversity distances: dispatcher, UniFrac, JSD, DPCoA, kind parameter."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyloseq
from pyloseq import (
    OtuTable,
    Phyloseq,
    SampleData,
    distance,
    distance_method_list,
    unifrac,
)

GOLDEN_DIR = Path("tests/golden")
ES_GOLDEN = GOLDEN_DIR / "esophagus"
ES_PRESENT = (ES_GOLDEN / "otu_table.parquet").exists()
UF_UN_PRESENT = (ES_GOLDEN / "unifrac_unweighted" / "normalized.parquet").exists()
UF_WT_PRESENT = (ES_GOLDEN / "unifrac_weighted" / "normalized.parquet").exists()


def _make_ps(ntaxa: int = 6, nsamples: int = 4) -> Phyloseq:
    rng = np.random.default_rng(42)
    counts = rng.integers(1, 100, size=(ntaxa, nsamples)).astype(float)
    taxa = [f"OTU{i + 1}" for i in range(ntaxa)]
    samples = [f"S{i + 1}" for i in range(nsamples)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    sam_df = pd.DataFrame(
        {"Group": ["A", "A", "B", "B"][:nsamples]},
        index=samples,
    )
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        sam=SampleData(sam_df),
    )


def _make_ps_with_tree() -> Phyloseq:
    from io import StringIO

    import skbio.tree

    from pyloseq import PhyTree

    newick = "((OTU1:0.1,OTU2:0.2):0.1,(OTU3:0.15,OTU4:0.05):0.2);"
    tree_node = skbio.tree.TreeNode.read(
        StringIO(newick), format="newick", convert_underscores=False
    )
    rng = np.random.default_rng(7)
    counts = rng.integers(1, 200, size=(4, 5)).astype(float)
    taxa = ["OTU1", "OTU2", "OTU3", "OTU4"]
    samples = [f"S{i + 1}" for i in range(5)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        tree=PhyTree(tree_node),
    )


def _load_esophagus() -> Phyloseq:
    from pyloseq import PhyTree
    from pyloseq.testing.fixtures import load_esophagus_reference

    ref = load_esophagus_reference()
    tree = PhyTree.from_newick(ref["phy_tree_newick"]) if "phy_tree_newick" in ref else None
    return Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tree=tree,
    )


# ===========================================================================
# distance dispatcher
# ===========================================================================


class TestDistance:
    def test_distance_method_list_returns_dict(self) -> None:
        ml = distance_method_list()
        assert isinstance(ml, dict)
        assert "vegan-equivalent" in ml
        assert "phylogenetic" in ml

    def test_bray_returns_dm(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps()
        dm = distance(ps, "bray")
        assert isinstance(dm, DistanceMatrix)
        assert len(dm.ids) == ps.nsamples

    def test_euclidean(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps()
        dm = distance(ps, "euclidean")
        assert isinstance(dm, DistanceMatrix)

    def test_jaccard_binary(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "jaccard")
        arr = np.array(dm.data)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0 + 1e-12

    def test_jsd(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps()
        dm = distance(ps, "jsd")
        assert isinstance(dm, DistanceMatrix)

    def test_unknown_method_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(pyloseq.pyloseqValidationError):
            distance(ps, "not_a_method")

    def test_symmetric(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "bray")
        arr = np.array(dm.data)
        np.testing.assert_allclose(arr, arr.T, atol=1e-12)

    def test_zero_diagonal(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "bray")
        arr = np.array(dm.data)
        np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)


# ===========================================================================
# UniFrac
# ===========================================================================


class TestUnifrac:
    def test_unifrac_unweighted(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps_with_tree()
        dm = unifrac(ps, weighted=False)
        assert isinstance(dm, DistanceMatrix)
        assert len(dm.ids) == ps.nsamples

    def test_unifrac_weighted(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps_with_tree()
        dm = unifrac(ps, weighted=True)
        assert isinstance(dm, DistanceMatrix)

    def test_unifrac_requires_tree(self) -> None:
        ps = _make_ps()
        with pytest.raises(pyloseq.pyloseqValidationError):
            unifrac(ps)

    def test_unifrac_symmetric(self) -> None:
        ps = _make_ps_with_tree()
        dm = unifrac(ps, weighted=False)
        arr = np.array(dm.data)
        np.testing.assert_allclose(arr, arr.T, atol=1e-12)

    def test_unifrac_zero_diagonal(self) -> None:
        ps = _make_ps_with_tree()
        dm = unifrac(ps, weighted=False)
        arr = np.array(dm.data)
        np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)

    def test_unifrac_via_distance(self) -> None:
        ps = _make_ps_with_tree()
        dm1 = unifrac(ps, weighted=False)
        dm2 = distance(ps, "unifrac")
        np.testing.assert_allclose(np.array(dm1.data), np.array(dm2.data), atol=1e-12)

    @pytest.mark.skipif(not (ES_PRESENT and UF_UN_PRESENT), reason="golden files not generated yet")
    def test_unweighted_unifrac_matches_r_esophagus(self) -> None:
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

    @pytest.mark.skipif(not (ES_PRESENT and UF_WT_PRESENT), reason="golden files not generated yet")
    def test_weighted_unifrac_matches_r_esophagus(self) -> None:
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


class TestJSD:
    def test_jsd_in_unit_range(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "jsd")
        arr = np.array(dm.data)
        assert arr.min() >= 0.0 - 1e-12
        assert arr.max() <= 1.0 + 1e-12

    def test_jsd_identical_samples_zero(self) -> None:
        df = pd.DataFrame(
            {"S1": [10.0, 20.0, 5.0], "S2": [10.0, 20.0, 5.0]}, index=["A", "B", "C"]
        )
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
        dm = distance(ps, "jsd")
        np.testing.assert_allclose(dm["S1", "S2"], 0.0, atol=1e-12)

    def test_jsd_completely_disjoint_is_one(self) -> None:
        df = pd.DataFrame({"S1": [10.0, 0.0], "S2": [0.0, 10.0]}, index=["A", "B"])
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
        dm = distance(ps, "jsd")
        np.testing.assert_allclose(dm["S1", "S2"], 1.0, atol=1e-10)


# ===========================================================================
# DPCoA returns a valid DistanceMatrix
# ===========================================================================


class TestDPCoA:
    def test_dpcoa_returns_dm(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps_with_tree()
        dm = distance(ps, "dpcoa")
        assert isinstance(dm, DistanceMatrix)
        arr = np.array(dm.data)
        np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)
        np.testing.assert_allclose(arr, arr.T, atol=1e-10)


# ===========================================================================
# 'kind' parameter and deprecated 'type' alias
# ===========================================================================


class TestDistanceKind:
    def test_distance_kind_parameter(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "bray", kind="samples")
        assert len(dm.ids) == ps.nsamples

    def test_distance_type_deprecated(self) -> None:
        ps = _make_ps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            distance(ps, "bray", type="samples")
            assert any("deprecated" in str(x.message).lower() for x in w)
