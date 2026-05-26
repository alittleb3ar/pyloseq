"""Tests for core container classes: OtuTable, SampleData, TaxTable, RefSeq, PhyTree."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from pyloseq import (
    OtuTable,
    PhyTree,
    RefSeq,
    SampleData,
    TaxTable,
)

GOLDEN_PRESENT = Path("tests/golden/GlobalPatterns/otu_table.parquet").exists()
skip_no_golden = pytest.mark.skipif(
    not GOLDEN_PRESENT,
    reason="golden files not generated — run `Rscript scripts/generate_golden.R`",
)


# ===========================================================================
# OtuTable
# ===========================================================================


class TestOtuTable:
    def test_construct_from_dataframe(self) -> None:
        df = pd.DataFrame([[1, 2], [3, 4]], index=["OTU1", "OTU2"], columns=["S1", "S2"])
        ot = OtuTable(df, taxa_are_rows=True)
        assert ot.ntaxa == 2
        assert ot.nsamples == 2
        assert list(ot.taxa_names) == ["OTU1", "OTU2"]
        assert list(ot.sample_names) == ["S1", "S2"]

    def test_construct_from_ndarray(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ot = OtuTable(arr, taxa_are_rows=True)
        assert ot.ntaxa == 2
        assert ot.nsamples == 2

    def test_construct_from_list(self) -> None:
        ot = OtuTable([[1, 2], [3, 4]])
        assert ot.ntaxa == 2

    def test_construct_from_sparse(self) -> None:
        mat = sp.csr_matrix(np.array([[1, 0], [0, 4]]))
        ot = OtuTable(mat, taxa_are_rows=True)
        assert ot.ntaxa == 2
        assert ot._is_sparse

    def test_orientation_flip_preserves_values(self) -> None:
        df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            index=["OTU1", "OTU2"],
            columns=["S1", "S2", "S3"],
        )
        ot = OtuTable(df, taxa_are_rows=True)
        original_values = ot._to_numpy().copy()

        ot.taxa_are_rows = False
        ot.taxa_are_rows = True  # flip back

        np.testing.assert_array_equal(ot._to_numpy(), original_values)

    def test_orientation_flip_preserves_logical_counts(self) -> None:
        df = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
        ot = OtuTable(df, taxa_are_rows=True)
        assert ot.ntaxa == 1
        assert ot.nsamples == 2
        assert list(ot.taxa_names) == ["OTU1"]
        assert list(ot.sample_names) == ["S1", "S2"]

        ot.taxa_are_rows = False
        assert ot.ntaxa == 1
        assert ot.nsamples == 2
        assert list(ot.taxa_names) == ["OTU1"]
        assert list(ot.sample_names) == ["S1", "S2"]

    def test_taxa_sums(self) -> None:
        df = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=["OTU1", "OTU2"],
            columns=["S1", "S2"],
        )
        ot = OtuTable(df, taxa_are_rows=True)
        sums = ot.taxa_sums()
        assert sums["OTU1"] == 30
        assert sums["OTU2"] == 70

    def test_sample_sums(self) -> None:
        df = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=["OTU1", "OTU2"],
            columns=["S1", "S2"],
        )
        ot = OtuTable(df, taxa_are_rows=True)
        sums = ot.sample_sums()
        assert sums["S1"] == 40
        assert sums["S2"] == 60

    def test_sums_consistent_after_flip(self) -> None:
        df = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=["OTU1", "OTU2"],
            columns=["S1", "S2"],
        )
        ot = OtuTable(df, taxa_are_rows=True)
        taxa_sums_before = ot.taxa_sums()
        sample_sums_before = ot.sample_sums()

        ot.taxa_are_rows = False

        pd.testing.assert_series_equal(ot.taxa_sums(), taxa_sums_before, check_names=False)
        pd.testing.assert_series_equal(ot.sample_sums(), sample_sums_before, check_names=False)

    def test_sparse_large_low_density(self) -> None:
        import tracemalloc

        rng = np.random.default_rng(0)
        n_taxa, n_samples = 100_000, 1_000
        mat = sp.random(n_taxa, n_samples, density=0.01, format="csr", random_state=rng)
        tracemalloc.start()
        ot = OtuTable(mat, taxa_are_rows=True)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert ot._is_sparse
        assert peak < 50 * 1024 * 1024, f"Peak memory {peak / 1e6:.1f} MB exceeded 50 MB"

    @given(
        np_arrays(
            np.float64,
            st.tuples(st.integers(1, 10), st.integers(1, 10)),
            elements=st.floats(0, 1e6, allow_nan=False, allow_infinity=False),
        ),
        st.booleans(),
    )
    @settings(max_examples=50)
    def test_nsamples_ntaxa_product(self, arr: np.ndarray, taxa_are_rows: bool) -> None:
        ot = OtuTable(arr, taxa_are_rows=taxa_are_rows)
        assert ot.nsamples * ot.ntaxa == arr.size

    @skip_no_golden
    def test_taxa_sums_match_r_reference(self) -> None:
        from pyloseq.testing import load_global_patterns_reference

        ref = load_global_patterns_reference()
        ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
        python_sums = ot.taxa_sums().sort_index()
        r_sums = ref["taxa_sums"].sort_index()
        np.testing.assert_allclose(python_sums.values, r_sums.values, atol=1e-10, rtol=1e-12)

    @skip_no_golden
    def test_sample_sums_match_r_reference(self) -> None:
        from pyloseq.testing import load_global_patterns_reference

        ref = load_global_patterns_reference()
        ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
        python_sums = ot.sample_sums().sort_index()
        r_sums = ref["sample_sums"].sort_index()
        np.testing.assert_allclose(python_sums.values, r_sums.values, atol=1e-10, rtol=1e-12)


# ===========================================================================
# SampleData, TaxTable
# ===========================================================================


class TestSampleData:
    def test_basic_construction(self) -> None:
        df = pd.DataFrame({"age": [25, 30], "site": ["gut", "oral"]}, index=["S1", "S2"])
        sd = SampleData(df)
        assert len(sd) == 2
        assert list(sd.variables) == ["age", "site"]

    def test_names(self) -> None:
        df = pd.DataFrame({"x": [1]}, index=["MySample"])
        sd = SampleData(df)
        assert list(sd.names) == ["MySample"]

    def test_to_frame_is_copy(self) -> None:
        df = pd.DataFrame({"x": [1]}, index=["S1"])
        sd = SampleData(df)
        frame = sd.to_frame()
        frame["x"] = 999
        assert sd.to_frame()["x"].iloc[0] == 1

    def test_categorical_dtype_preserved(self) -> None:
        df = pd.DataFrame({"grp": pd.Categorical(["a", "b", "a"])}, index=["S1", "S2", "S3"])
        sd = SampleData(df)
        assert isinstance(sd.to_frame()["grp"].dtype, pd.CategoricalDtype)

    @skip_no_golden
    def test_enterotype_shape(self) -> None:
        from pyloseq.testing import load_enterotype_reference

        ref = load_enterotype_reference()
        sd = SampleData(ref["sample_data"])
        assert sd.to_frame().shape == (280, 9)


class TestTaxTable:
    def test_basic_construction(self) -> None:
        df = pd.DataFrame({"Phylum": ["Firmicutes"], "Genus": ["Bacillus"]}, index=["OTU1"])
        tt = TaxTable(df)
        assert tt.rank_names == ["Phylum", "Genus"]
        assert list(tt.names) == ["OTU1"]

    def test_to_frame_is_copy(self) -> None:
        df = pd.DataFrame({"Phylum": ["Firmicutes"]}, index=["OTU1"])
        tt = TaxTable(df)
        frame = tt.to_frame()
        frame["Phylum"] = "Other"
        assert tt.to_frame()["Phylum"].iloc[0] == "Firmicutes"

    @skip_no_golden
    def test_global_patterns_shape(self) -> None:
        from pyloseq.testing import load_global_patterns_reference

        ref = load_global_patterns_reference()
        tt = TaxTable(ref["tax_table"])
        assert tt.to_frame().shape == (19216, 7)


# ===========================================================================
# RefSeq
# ===========================================================================


class TestRefSeq:
    def test_from_fasta_round_trip(self, tmp_path: Path) -> None:
        import skbio

        seqs = {
            "OTU1": skbio.DNA("ACGT", metadata={"id": "OTU1", "description": ""}),
            "OTU2": skbio.DNA("TTTT", metadata={"id": "OTU2", "description": ""}),
        }
        rs = RefSeq(seqs)
        fasta = tmp_path / "seqs.fasta"
        rs.to_fasta(fasta)
        rs2 = RefSeq.from_fasta(fasta)
        assert set(rs2.names) == {"OTU1", "OTU2"}
        assert str(rs2["OTU1"]) == "ACGT"
        assert str(rs2["OTU2"]) == "TTTT"

    def test_copy_is_independent(self) -> None:
        import skbio

        rs = RefSeq({"OTU1": skbio.DNA("ACGT")})
        rs2 = rs.copy()
        rs._seqs["OTU2"] = skbio.DNA("TTTT")
        assert "OTU2" not in rs2.names


# ===========================================================================
# PhyTree
# ===========================================================================


class TestPhyTree:
    SIMPLE_NWK = "((OTU1:0.1,OTU2:0.2):0.3,OTU3:0.4);"

    def test_from_newick(self) -> None:
        t = PhyTree.from_newick(self.SIMPLE_NWK)
        assert t.n_tips == 3
        assert set(t.tip_names) == {"OTU1", "OTU2", "OTU3"}

    def test_total_branch_length(self) -> None:
        t = PhyTree.from_newick(self.SIMPLE_NWK)
        assert abs(t.total_branch_length - 1.0) < 1e-12

    def test_is_rooted(self) -> None:
        rooted = PhyTree.from_newick("(A:1.0,B:1.0);")
        assert rooted.is_rooted
        unrooted = PhyTree.from_newick("(A:1.0,B:1.0,C:1.0);")
        assert not unrooted.is_rooted

    def test_prune(self) -> None:
        t = PhyTree.from_newick(self.SIMPLE_NWK)
        pruned = t.prune(["OTU1", "OTU2"])
        assert pruned.n_tips == 2
        assert set(pruned.tip_names) == {"OTU1", "OTU2"}
        assert pruned.total_branch_length < t.total_branch_length

    def test_newick_round_trip(self) -> None:
        t = PhyTree.from_newick(self.SIMPLE_NWK)
        nwk2 = t.to_newick()
        t2 = PhyTree.from_newick(nwk2)
        np.testing.assert_allclose(t.total_branch_length, t2.total_branch_length, atol=1e-12)

    @skip_no_golden
    def test_global_patterns_tree_ntips(self) -> None:
        from pyloseq.testing import load_global_patterns_reference

        ref = load_global_patterns_reference()
        t = PhyTree.from_newick(ref["phy_tree_newick"])
        assert t.n_tips == 19216

    @skip_no_golden
    def test_esophagus_tree_ntips(self) -> None:
        from pyloseq.testing import load_esophagus_reference

        ref = load_esophagus_reference()
        t = PhyTree.from_newick(ref["phy_tree_newick"])
        assert t.n_tips == 58

    def test_eq_different_tip_sets(self) -> None:
        t1 = PhyTree.from_newick("(A:0.1,B:0.2);")
        t2 = PhyTree.from_newick("(A:0.1,C:0.2);")
        assert t1 != t2

    def test_eq_same_newick(self) -> None:
        nwk = "((A:0.1,B:0.2):0.05,C:0.3);"
        assert PhyTree.from_newick(nwk) == PhyTree.from_newick(nwk)
