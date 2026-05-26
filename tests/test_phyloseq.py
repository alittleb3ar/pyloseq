"""Tests for the Phyloseq constructor, validators, property setters, and accessors."""

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


def _make_simple_ps(n_taxa: int = 3, n_samples: int = 2) -> Phyloseq:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.integers(0, 100, size=(n_taxa, n_samples)).astype(float),
        index=[f"OTU{i}" for i in range(n_taxa)],
        columns=[f"S{j}" for j in range(n_samples)],
    )
    return Phyloseq(otu=OtuTable(df))


# ===========================================================================
# Constructor and validators
# ===========================================================================


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

    def test_setter_warns_when_pruning_occurs(self) -> None:
        otu1 = OtuTable(
            pd.DataFrame({"S1": [1.0, 2.0]}, index=["OTU1", "OTU2"]), taxa_are_rows=True
        )
        tax = TaxTable(pd.DataFrame({"Phylum": ["A", "B"]}, index=["OTU1", "OTU2"]))
        ps = Phyloseq(otu=otu1, tax=tax)
        new_otu = OtuTable(pd.DataFrame({"S1": [1.0]}, index=["OTU1"]), taxa_are_rows=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps.otu_table = new_otu
            assert any("pruned" in str(x.message).lower() for x in w)
        assert ps.tax_table is not None
        assert "OTU2" not in ps.tax_table.names

    def test_tree_with_extra_tips_does_not_prune_otu(self) -> None:
        newick = "((OTU1:0.1,OTU2:0.2):0.1,(OTU3:0.15,OTU4:0.05):0.2);"
        tree = PhyTree.from_newick(newick)
        df = pd.DataFrame({"S1": [1.0, 2.0]}, index=["OTU1", "OTU2"])
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), tree=tree)
        assert ps.ntaxa == 2
        assert ps.phy_tree is not None
        assert set(ps.phy_tree.tip_names) == {"OTU1", "OTU2"}

    def test_tree_subset_of_taxa_does_not_shrink_otu(self) -> None:
        newick = "(OTU1:0.1,OTU2:0.2);"
        tree = PhyTree.from_newick(newick)
        df = pd.DataFrame({"S1": [1.0, 2.0, 3.0]}, index=["OTU1", "OTU2", "OTU3"])
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), tree=tree)
        assert "OTU3" in ps.taxa_names

    def test_validate_does_not_compute_sample_sets_twice(self) -> None:
        otu = OtuTable(pd.DataFrame({"S1": [1.0]}, index=["OTU1"]), taxa_are_rows=True)
        sam = SampleData(pd.DataFrame({"x": [1]}, index=["S999"]))
        with pytest.raises(pyloseqValidationError, match="sample names"):
            Phyloseq(otu=otu, sam=sam)

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
        assert len(ps.sample_variables) == 7

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
# Accessors
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
