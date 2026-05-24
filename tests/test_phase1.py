"""Phase 1 tests: core container and validators.

Golden-file tests are skipped when tests/golden/ hasn't been populated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

import pyloseq
from pyloseq import (
    OtuTable,
    Phyloseq,
    PhyTree,
    RefSeq,
    SampleData,
    TaxTable,
    pyloseqValidationError,
)

GOLDEN_PRESENT = Path("tests/golden/GlobalPatterns/otu_table.parquet").exists()
skip_no_golden = pytest.mark.skipif(
    not GOLDEN_PRESENT,
    reason="golden files not generated — run `Rscript scripts/generate_golden.R`",
)


# ===========================================================================
# Ticket 1.1 — OtuTable
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
        # Flipping taxa_are_rows transposes the stored matrix but the number
        # of taxa and samples is a property of the data, not the orientation.
        df = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
        ot = OtuTable(df, taxa_are_rows=True)
        assert ot.ntaxa == 1
        assert ot.nsamples == 2
        assert list(ot.taxa_names) == ["OTU1"]
        assert list(ot.sample_names) == ["S1", "S2"]

        ot.taxa_are_rows = False
        # Counts and logical identity unchanged after orientation flip
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
        # 1% density → 1M nonzero elements
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
# Ticket 1.2 — SampleData, TaxTable, RefSeq
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


# ===========================================================================
# Ticket 1.3 — PhyTree
# ===========================================================================


class TestPhyTree:
    SIMPLE_NWK = "((OTU1:0.1,OTU2:0.2):0.3,OTU3:0.4);"

    def test_from_newick(self) -> None:
        t = PhyTree.from_newick(self.SIMPLE_NWK)
        assert t.n_tips == 3
        assert set(t.tip_names) == {"OTU1", "OTU2", "OTU3"}

    def test_total_branch_length(self) -> None:
        t = PhyTree.from_newick(self.SIMPLE_NWK)
        # 0.1 + 0.2 + 0.3 + 0.4 = 1.0
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


# ===========================================================================
# Ticket 1.4 + 1.5 — Phyloseq constructor and validators
# ===========================================================================


def _make_simple_ps(n_taxa: int = 3, n_samples: int = 2) -> Phyloseq:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.integers(0, 100, size=(n_taxa, n_samples)).astype(float),
        index=[f"OTU{i}" for i in range(n_taxa)],
        columns=[f"S{j}" for j in range(n_samples)],
    )
    return Phyloseq(otu=OtuTable(df))


class TestPhyloseqConstructor:
    def test_otu_only(self) -> None:
        ps = _make_simple_ps()
        assert ps.ntaxa == 3
        assert ps.nsamples == 2
        assert ps.sample_data is None
        assert ps.tax_table is None

    def test_with_sample_data(self) -> None:
        df_otu = pd.DataFrame([[1, 2], [3, 4]], index=["OTU1", "OTU2"], columns=["S1", "S2"])
        df_sam = pd.DataFrame({"group": ["A", "B"]}, index=["S1", "S2"])
        ps = Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam))
        assert ps.nsamples == 2
        assert ps.sample_variables == ["group"]

    def test_mismatched_taxa_pruned_to_intersection(self) -> None:
        df_otu = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=["A", "B", "C"],
            columns=["S1", "S2"],
        )
        df_tax = pd.DataFrame({"Phylum": ["P1", "P2"]}, index=["A", "B"])
        ps = Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax))
        # C only in OTU table → pruned out
        assert ps.ntaxa == 2
        assert "C" not in ps.taxa_names

    def test_mismatched_samples_pruned_to_intersection(self) -> None:
        df_otu = pd.DataFrame([[1, 2, 3]], index=["OTU1"], columns=["S1", "S2", "S3"])
        df_sam = pd.DataFrame({"x": [1, 2]}, index=["S1", "S2"])
        ps = Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam))
        assert ps.nsamples == 2
        assert "S3" not in ps.sample_names

    def test_strict_mode_raises_on_taxa_mismatch(self) -> None:
        df_otu = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["S1", "S2"])
        df_tax = pd.DataFrame({"Phylum": ["P1"]}, index=["A"])
        with pytest.raises(pyloseqValidationError):
            Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax), strict=True)

    def test_strict_mode_raises_on_sample_mismatch(self) -> None:
        df_otu = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
        df_sam = pd.DataFrame({"x": [1]}, index=["S1"])
        with pytest.raises(pyloseqValidationError):
            Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam), strict=True)

    def test_missing_otu_table_raises(self) -> None:
        with pytest.raises((TypeError, pyloseqValidationError)):
            Phyloseq(otu=None)  # type: ignore[arg-type]

    def test_empty_taxa_intersection_raises(self) -> None:
        df_otu = pd.DataFrame([[1]], index=["OTU_X"], columns=["S1"])
        df_tax = pd.DataFrame({"Phylum": ["P1"]}, index=["OTU_Y"])
        with pytest.raises(pyloseqValidationError, match="taxa/OTU names do not match"):
            Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax))

    def test_empty_sample_intersection_raises(self) -> None:
        df_otu = pd.DataFrame([[1]], index=["OTU1"], columns=["S_X"])
        df_sam = pd.DataFrame({"x": [1]}, index=["S_Y"])
        with pytest.raises(pyloseqValidationError, match="sample names do not match"):
            Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam))

    def test_sample_data_setter_reruns_validation(self) -> None:
        ps = _make_simple_ps(n_taxa=3, n_samples=2)
        smaller = SampleData(pd.DataFrame({"x": [1]}, index=["S0"]))
        ps.sample_data = smaller
        assert ps.nsamples == 1

    @skip_no_golden
    def test_global_patterns_dimensions(self) -> None:
        from pyloseq.testing import load_global_patterns_reference

        ref = load_global_patterns_reference()
        ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
        tt = TaxTable(ref["tax_table"])
        sd = SampleData(ref["sample_data"])
        tree = PhyTree.from_newick(ref["phy_tree_newick"])
        ps = Phyloseq(otu=ot, sam=sd, tax=tt, tree=tree)
        assert ps.ntaxa == 19216
        assert ps.nsamples == 26
        assert len(ps.rank_names) == 7
        assert len(ps.sample_variables) == 7  # GlobalPatterns has 7 sample variables

    @skip_no_golden
    def test_esophagus_dimensions(self) -> None:
        from pyloseq.testing import load_esophagus_reference

        ref = load_esophagus_reference()
        ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
        tree = PhyTree.from_newick(ref["phy_tree_newick"])
        ps = Phyloseq(otu=ot, tree=tree)
        assert ps.ntaxa == 58
        assert ps.nsamples == 3


# ===========================================================================
# Ticket 1.6 — Accessors
# ===========================================================================


class TestAccessors:
    def setup_method(self) -> None:
        df_otu = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=["OTU1", "OTU2"],
            columns=["S1", "S2"],
        )
        df_sam = pd.DataFrame({"group": ["A", "B"], "depth": [100, 200]}, index=["S1", "S2"])
        df_tax = pd.DataFrame(
            {"Phylum": ["Firm", "Bact"], "Genus": ["Lacto", "Bact"]},
            index=["OTU1", "OTU2"],
        )
        self.ps = Phyloseq(
            otu=OtuTable(df_otu),
            sam=SampleData(df_sam),
            tax=TaxTable(df_tax),
        )

    def test_taxa_names(self) -> None:
        assert set(self.ps.taxa_names) == {"OTU1", "OTU2"}

    def test_sample_names(self) -> None:
        assert set(self.ps.sample_names) == {"S1", "S2"}

    def test_ntaxa_nsamples(self) -> None:
        assert self.ps.ntaxa == 2
        assert self.ps.nsamples == 2

    def test_sample_variables(self) -> None:
        assert self.ps.sample_variables == ["group", "depth"]

    def test_rank_names(self) -> None:
        assert self.ps.rank_names == ["Phylum", "Genus"]

    def test_get_variable(self) -> None:
        s = self.ps.get_variable("group")
        assert isinstance(s, pd.Series)
        assert list(s) == ["A", "B"]

    def test_get_taxa(self) -> None:
        vec = self.ps.get_taxa("OTU1")
        assert isinstance(vec, pd.Series)
        assert vec["S1"] == 10
        assert vec["S2"] == 20

    def test_get_sample(self) -> None:
        vec = self.ps.get_sample("S1")
        assert isinstance(vec, pd.Series)
        assert vec["OTU1"] == 10
        assert vec["OTU2"] == 30

    def test_taxa_sums(self) -> None:
        sums = self.ps.taxa_sums()
        assert isinstance(sums, pd.Series)
        assert sums["OTU1"] == 30
        assert sums["OTU2"] == 70

    def test_sample_sums(self) -> None:
        sums = self.ps.sample_sums()
        assert isinstance(sums, pd.Series)
        assert sums["S1"] == 40
        assert sums["S2"] == 60

    def test_repr_contains_dimensions(self) -> None:
        r = repr(self.ps)
        assert "2 taxa" in r
        assert "2 samples" in r

    def test_public_api_exports(self) -> None:
        for name in ["Phyloseq", "OtuTable", "SampleData", "TaxTable", "PhyTree", "RefSeq"]:
            assert hasattr(pyloseq, name)
