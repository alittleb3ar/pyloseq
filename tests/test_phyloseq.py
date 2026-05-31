"""Tests for the Phyloseq constructor, validators, property setters, and accessors."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import pyloseq
from pyloseq import (
    OtuTable,
    Phyloseq,
    PhyTree,
    SampleData,
    TaxTable,
    pyloseqValidationError,
)
from pyloseq.datasets import load_esophagus_reference, load_global_patterns_reference

# ===========================================================================
# Constructor and validators
# ===========================================================================


def test_otu_only(simple_ps: Phyloseq) -> None:
    assert simple_ps.ntaxa == 3
    assert simple_ps.nsamples == 2
    assert simple_ps.sample_data is None
    assert simple_ps.tax_table is None


def test_with_sample_data() -> None:
    df_otu = pd.DataFrame(
        [[1, 2], [3, 4]], index=["OTU1", "OTU2"], columns=["S1", "S2"]
    )
    df_sam = pd.DataFrame({"group": ["A", "B"]}, index=["S1", "S2"])
    ps = Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam))
    assert ps.nsamples == 2
    assert ps.sample_variables == ["group"]


def test_mismatched_taxa_pruned_to_intersection() -> None:
    df_otu = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=["A", "B", "C"],
        columns=["S1", "S2"],
    )
    df_tax = pd.DataFrame({"Phylum": ["P1", "P2"]}, index=["A", "B"])
    ps = Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax))
    assert ps.ntaxa == 2
    assert "C" not in ps.taxa_names


def test_mismatched_samples_pruned_to_intersection() -> None:
    df_otu = pd.DataFrame([[1, 2, 3]], index=["OTU1"], columns=["S1", "S2", "S3"])
    df_sam = pd.DataFrame({"x": [1, 2]}, index=["S1", "S2"])
    ps = Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam))
    assert ps.nsamples == 2
    assert "S3" not in ps.sample_names


def test_strict_mode_raises_on_taxa_mismatch() -> None:
    df_otu = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["S1", "S2"])
    df_tax = pd.DataFrame({"Phylum": ["P1"]}, index=["A"])
    with pytest.raises(pyloseqValidationError, match="taxa/OTU names do not match"):
        Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax), strict=True)


def test_strict_mode_raises_on_sample_mismatch() -> None:
    df_otu = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
    df_sam = pd.DataFrame({"x": [1]}, index=["S1"])
    with pytest.raises(pyloseqValidationError, match="sample names do not match"):
        Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam), strict=True)


def test_missing_otu_table_raises() -> None:
    with pytest.raises(pyloseqValidationError):
        Phyloseq(otu=None)  # type: ignore[arg-type]


def test_empty_taxa_intersection_raises() -> None:
    df_otu = pd.DataFrame([[1]], index=["OTU_X"], columns=["S1"])
    df_tax = pd.DataFrame({"Phylum": ["P1"]}, index=["OTU_Y"])
    with pytest.raises(pyloseqValidationError, match="taxa/OTU names do not match"):
        Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax))


def test_empty_sample_intersection_raises() -> None:
    df_otu = pd.DataFrame([[1]], index=["OTU1"], columns=["S_X"])
    df_sam = pd.DataFrame({"x": [1]}, index=["S_Y"])
    with pytest.raises(pyloseqValidationError, match="sample names do not match"):
        Phyloseq(otu=OtuTable(df_otu), sam=SampleData(df_sam))


def test_sample_data_setter_reruns_validation(simple_ps: Phyloseq) -> None:
    smaller = SampleData(pd.DataFrame({"x": [1]}, index=["S0"]))
    simple_ps.sample_data = smaller
    assert simple_ps.nsamples == 1


def test_setter_warns_when_pruning_occurs() -> None:
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
    assert "OTU2" not in ps.tax_table.taxa_names


def test_tree_with_extra_tips_does_not_prune_otu() -> None:
    newick = "((OTU1:0.1,OTU2:0.2):0.1,(OTU3:0.15,OTU4:0.05):0.2);"
    tree = PhyTree.from_newick(newick)
    df = pd.DataFrame({"S1": [1.0, 2.0]}, index=["OTU1", "OTU2"])
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), tree=tree)
    assert ps.ntaxa == 2
    assert ps.phy_tree is not None
    assert set(ps.phy_tree.tip_names) == {"OTU1", "OTU2"}


def test_tree_subset_of_taxa_does_not_shrink_otu() -> None:
    newick = "(OTU1:0.1,OTU2:0.2);"
    tree = PhyTree.from_newick(newick)
    df = pd.DataFrame({"S1": [1.0, 2.0, 3.0]}, index=["OTU1", "OTU2", "OTU3"])
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), tree=tree)
    assert "OTU3" in ps.taxa_names


def test_global_patterns_dimensions() -> None:

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


def test_esophagus_dimensions() -> None:

    ref = load_esophagus_reference()
    ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
    tree = PhyTree.from_newick(ref["phy_tree_newick"])
    ps = Phyloseq(otu=ot, tree=tree)
    assert ps.ntaxa == 58
    assert ps.nsamples == 3


# ===========================================================================
# Accessors
# ===========================================================================


def test_taxa_names(full_ps: Phyloseq) -> None:
    assert set(full_ps.taxa_names) == {"OTU1", "OTU2"}


def test_sample_names(full_ps: Phyloseq) -> None:
    assert set(full_ps.sample_names) == {"S1", "S2"}


def test_ntaxa_nsamples(full_ps: Phyloseq) -> None:
    assert full_ps.ntaxa == 2
    assert full_ps.nsamples == 2


def test_sample_variables(full_ps: Phyloseq) -> None:
    assert full_ps.sample_variables == ["group", "depth"]


def test_rank_names(full_ps: Phyloseq) -> None:
    assert full_ps.rank_names == ["Phylum", "Genus"]


def test_get_variable(full_ps: Phyloseq) -> None:
    s = full_ps.get_variable("group")
    assert isinstance(s, pd.Series)
    assert list(s) == ["A", "B"]


def test_get_taxa(full_ps: Phyloseq) -> None:
    vec = full_ps.get_taxa("OTU1")
    assert isinstance(vec, pd.Series)
    assert vec["S1"] == 10
    assert vec["S2"] == 20


def test_get_sample(full_ps: Phyloseq) -> None:
    vec = full_ps.get_sample("S1")
    assert isinstance(vec, pd.Series)
    assert vec["OTU1"] == 10
    assert vec["OTU2"] == 30


def test_taxa_sums(full_ps: Phyloseq) -> None:
    sums = full_ps.taxa_sums()
    assert isinstance(sums, pd.Series)
    assert sums["OTU1"] == 30
    assert sums["OTU2"] == 70


def test_sample_sums(full_ps: Phyloseq) -> None:
    sums = full_ps.sample_sums()
    assert isinstance(sums, pd.Series)
    assert sums["S1"] == 40
    assert sums["S2"] == 60


def test_repr_contains_dimensions(full_ps: Phyloseq) -> None:
    r = repr(full_ps)
    assert "2 taxa" in r
    assert "2 samples" in r


def test_public_api_exports() -> None:
    for name in [
        "Phyloseq",
        "OtuTable",
        "SampleData",
        "TaxTable",
        "PhyTree",
        "RefSeq",
    ]:
        assert hasattr(pyloseq, name)


# ===========================================================================
# refseq, quiet, metadata
# ===========================================================================


def test_refseq_component_stored_and_accessible() -> None:
    import skbio

    from pyloseq import RefSeq

    df = pd.DataFrame({"S1": [1.0, 2.0]}, index=["OTU1", "OTU2"])
    rs = RefSeq({"OTU1": skbio.DNA("ACGT"), "OTU2": skbio.DNA("TTTT")})
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), refseq=rs)
    assert ps.refseq is not None
    assert set(ps.refseq.taxa_names) == {"OTU1", "OTU2"}


def test_quiet_suppresses_pruning_warning() -> None:
    df_otu = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]], index=["A", "B", "C"], columns=["S1", "S2"]
    )
    df_tax = pd.DataFrame({"Phylum": ["P1", "P2"]}, index=["A", "B"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Phyloseq(otu=OtuTable(df_otu), tax=TaxTable(df_tax), quiet=True)
    assert len(w) == 0


def test_metadata_attribute_stored_and_accessible() -> None:
    df = pd.DataFrame({"S1": [1.0]}, index=["OTU1"])
    ps = Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        metadata={"source": "test", "version": 2},
    )
    assert ps.metadata["source"] == "test"
    assert ps.metadata["version"] == 2


def test_metadata_defaults_to_empty_dict() -> None:
    df = pd.DataFrame({"S1": [1.0]}, index=["OTU1"])
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
    assert ps.metadata == {}
