"""Tests for tax_glom, tip_glom, merge_taxa, merge_phyloseq, merge_samples, psmelt."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skbio
import skbio.tree

import pyloseq
from pyloseq import (OtuTable, Phyloseq, PhyTree, RefSeq, SampleData, TaxTable,
                     merge_phyloseq, merge_samples, merge_taxa, psmelt,
                     tax_glom, tip_glom)
from pyloseq.datasets.fixtures import load_global_patterns_reference

GOLDEN_DIR = Path("tests/golden")
GP_GOLDEN = GOLDEN_DIR / "GlobalPatterns"
GP_TAXGLOM_PRESENT = (GP_GOLDEN / "tax_glom_Family" / "taxa_sums.parquet").exists()
GP_MERGESAM_PRESENT = (
    GP_GOLDEN / "merge_samples_SampleType" / "sample_sums.parquet"
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
# tax_glom
# ===========================================================================


def test_tax_glom_collapses_taxa(ps: Phyloseq) -> None:
    ps2 = tax_glom(ps, "Phylum")
    assert ps2.ntaxa == 4


def test_tax_glom_sums_abundances(ps: Phyloseq) -> None:
    ps2 = tax_glom(ps, "Phylum")
    orig_total = ps.taxa_sums().sum()
    assert abs(ps2.taxa_sums().sum() - orig_total) < 1e-10


def test_tax_glom_at_genus_gives_more_groups(ps: Phyloseq) -> None:
    ps_phylum = tax_glom(ps, "Phylum")
    ps_genus = tax_glom(ps, "Genus")
    assert ps_genus.ntaxa >= ps_phylum.ntaxa


def test_tax_glom_no_tax_raises() -> None:
    ps = _make_ps(with_tax=False)
    with pytest.raises(pyloseq.pyloseqValidationError):
        tax_glom(ps, "Phylum")


def test_tax_glom_bad_rank_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        tax_glom(ps, "Species")


def test_tax_glom_preserves_sample_data(ps: Phyloseq) -> None:
    ps2 = tax_glom(ps, "Phylum")
    assert ps2.sample_data is not None
    assert ps2.nsamples == ps.nsamples


def test_tax_glom_na_rm(ps: Phyloseq) -> None:
    tax_df = ps.tax_table.to_frame().copy()
    tax_df.loc["OTU1", "Phylum"] = ""
    ps2 = Phyloseq(
        otu=ps.otu_table.copy(),
        tax=TaxTable(tax_df),
    )
    ps3 = tax_glom(ps2, "Phylum", na_rm=True)
    assert ps3.ntaxa < ps.ntaxa


def test_tax_glom_preserves_refseq() -> None:
    df = pd.DataFrame({"S1": [5.0, 3.0]}, index=["OTU1", "OTU2"])
    tax_df = pd.DataFrame(
        {"Phylum": ["Firmicutes", "Firmicutes"]}, index=["OTU1", "OTU2"]
    )
    rs = RefSeq({"OTU1": skbio.DNA("ACGT"), "OTU2": skbio.DNA("TTTT")})
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), tax=TaxTable(tax_df), refseq=rs)
    ps2 = tax_glom(ps, "Phylum")
    assert ps2.refseq is not None
    assert len(ps2.refseq) == 1


@pytest.mark.skipif(not GP_TAXGLOM_PRESENT, reason="golden files not generated yet")
def test_tax_glom_family_matches_r() -> None:
    ref = load_global_patterns_reference()
    gp = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tax=TaxTable(ref["tax_table"]),
    )
    gp_fam = tax_glom(gp, "Family")

    golden_ts = pd.read_parquet(GP_GOLDEN / "tax_glom_Family" / "taxa_sums.parquet")
    if "__index__" in golden_ts.columns:
        golden_ts = golden_ts.set_index("__index__")
        golden_ts.index.name = None

    assert gp_fam.ntaxa == len(golden_ts)
    np.testing.assert_allclose(
        gp_fam.taxa_sums().sum(),
        golden_ts["value"].sum(),
        atol=1e-6,
    )


# ===========================================================================
# tip_glom
# ===========================================================================


def test_tip_glom_reduces_taxa() -> None:
    ps = _make_ps_with_tree()
    ps2 = tip_glom(ps, h=0.25)
    assert ps2.ntaxa <= ps.ntaxa


def test_tip_glom_no_tree_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        tip_glom(ps, h=0.1)


def test_tip_glom_large_h_merges_all() -> None:
    ps = _make_ps_with_tree()
    ps2 = tip_glom(ps, h=100.0)
    assert ps2.ntaxa == 1


def test_tip_glom_zero_h_keeps_all() -> None:
    ps = _make_ps_with_tree()
    ps2 = tip_glom(ps, h=0.0)
    assert ps2.ntaxa == ps.ntaxa


def test_tip_glom_result_has_fewer_or_equal_taxa() -> None:
    ps = _make_ps_with_tree()
    ps2 = tip_glom(ps, h=0.5)
    assert ps2.ntaxa <= ps.ntaxa


def test_tip_glom_abundance_conserved() -> None:
    ps = _make_ps_with_tree()
    total_before = ps.otu_table.to_dataframe().values.sum()
    ps2 = tip_glom(ps, h=0.5)
    total_after = ps2.otu_table.to_dataframe().values.sum()
    np.testing.assert_allclose(total_before, total_after, rtol=1e-10)


def test_tip_glom_hcfun_complete_reduces_taxa() -> None:
    ps = _make_ps_with_tree()
    ps2 = tip_glom(ps, h=0.25, hcfun="complete")
    assert ps2.ntaxa <= ps.ntaxa


def test_tip_glom_hcfun_complete_abundance_conserved() -> None:
    ps = _make_ps_with_tree()
    total_before = ps.otu_table.to_dataframe().values.sum()
    ps2 = tip_glom(ps, h=0.25, hcfun="complete")
    np.testing.assert_allclose(
        ps2.otu_table.to_dataframe().values.sum(), total_before, rtol=1e-10
    )


# ===========================================================================
# merge_taxa
# ===========================================================================


def test_merge_taxa_reduces_ntaxa(ps: Phyloseq) -> None:
    ps2 = merge_taxa(ps, ["OTU1", "OTU2"])
    assert ps2.ntaxa == ps.ntaxa - 1


def test_merge_taxa_sums_abundances(ps: Phyloseq) -> None:
    orig = ps.otu_table.to_dataframe()
    ps2 = merge_taxa(ps, ["OTU1", "OTU2"])
    result = ps2.otu_table.to_dataframe()
    archetype = (result.index.intersection(pd.Index(["OTU1", "OTU2"])))[0]
    np.testing.assert_allclose(
        result.loc[archetype].values,
        (orig.loc["OTU1"] + orig.loc["OTU2"]).values,
        atol=1e-10,
    )


def test_merge_taxa_explicit_archetype(ps: Phyloseq) -> None:
    ps2 = merge_taxa(ps, ["OTU1", "OTU2", "OTU3"], archetype="OTU2")
    assert "OTU2" in list(ps2.taxa_names)
    assert "OTU1" not in list(ps2.taxa_names)


def test_merge_taxa_single_taxon_noop(ps: Phyloseq) -> None:
    ps2 = merge_taxa(ps, ["OTU1"])
    assert ps2.ntaxa == ps.ntaxa


def test_merge_taxa_total_abundance_preserved(ps: Phyloseq) -> None:
    orig_total = ps.taxa_sums().sum()
    ps2 = merge_taxa(ps, ["OTU1", "OTU2", "OTU3"])
    assert abs(ps2.taxa_sums().sum() - orig_total) < 1e-10


def test_merge_taxa_preserves_refseq() -> None:
    ps = _make_ps_with_refseq()
    ps2 = merge_taxa(ps, ["OTU1", "OTU2"])
    assert ps2.refseq is not None


# ===========================================================================
# merge_phyloseq
# ===========================================================================


def test_merge_phyloseq_two_disjoint_samples() -> None:
    ps1 = _make_ps(nsamples=2, with_sam=False, with_tax=False)
    df2 = pd.DataFrame(
        {"S5": [1.0, 2.0], "S6": [3.0, 4.0]},
        index=["OTU1", "OTU2"],
    )
    ps2 = Phyloseq(otu=OtuTable(df2, taxa_are_rows=True))
    merged = merge_phyloseq(ps1, ps2)
    assert merged.nsamples == ps1.nsamples + ps2.nsamples


def test_merge_phyloseq_sums_overlapping_abundances() -> None:
    df1 = pd.DataFrame({"S1": [10.0, 5.0]}, index=["OTU1", "OTU2"])
    df2 = pd.DataFrame({"S1": [3.0, 7.0]}, index=["OTU1", "OTU2"])
    ps1 = Phyloseq(otu=OtuTable(df1, taxa_are_rows=True))
    ps2 = Phyloseq(otu=OtuTable(df2, taxa_are_rows=True))
    merged = merge_phyloseq(ps1, ps2)
    assert merged.ntaxa == 2
    assert merged.nsamples == 1
    np.testing.assert_allclose(
        merged.otu_table.to_dataframe()["S1"].values, [13.0, 12.0]
    )


def test_merge_phyloseq_union_taxa() -> None:
    df1 = pd.DataFrame({"S1": [10.0]}, index=["OTU1"])
    df2 = pd.DataFrame({"S2": [5.0]}, index=["OTU2"])
    ps1 = Phyloseq(otu=OtuTable(df1, taxa_are_rows=True))
    ps2 = Phyloseq(otu=OtuTable(df2, taxa_are_rows=True))
    merged = merge_phyloseq(ps1, ps2)
    assert merged.ntaxa == 2
    assert merged.nsamples == 2


def test_merge_phyloseq_requires_two_or_more(ps: Phyloseq) -> None:
    with pytest.raises(ValueError):
        merge_phyloseq(ps)


def test_merge_phyloseq_three_objects_sample_count() -> None:
    ps1 = _make_ps(nsamples=2, with_sam=False, with_tax=False)
    df2 = pd.DataFrame({"S5": [1.0, 2.0]}, index=["OTU1", "OTU2"])
    df3 = pd.DataFrame({"S6": [3.0, 4.0]}, index=["OTU1", "OTU2"])
    ps2 = Phyloseq(otu=OtuTable(df2, taxa_are_rows=True))
    ps3 = Phyloseq(otu=OtuTable(df3, taxa_are_rows=True))
    merged = merge_phyloseq(ps1, ps2, ps3)
    assert merged.nsamples == ps1.nsamples + 2


def test_merge_phyloseq_three_objects_abundance_sum() -> None:
    """All three objects share S1 — abundances should be summed across all three."""
    df1 = pd.DataFrame({"S1": [10.0]}, index=["OTU1"])
    df2 = pd.DataFrame({"S1": [5.0]}, index=["OTU1"])
    df3 = pd.DataFrame({"S1": [3.0]}, index=["OTU1"])
    merged = merge_phyloseq(
        Phyloseq(otu=OtuTable(df1, taxa_are_rows=True)),
        Phyloseq(otu=OtuTable(df2, taxa_are_rows=True)),
        Phyloseq(otu=OtuTable(df3, taxa_are_rows=True)),
    )
    otu_df = merged.otu_table.to_dataframe()
    np.testing.assert_allclose(float(otu_df.loc["OTU1", "S1"]), 18.0, atol=1e-10)


# ===========================================================================
# merge_samples
# ===========================================================================


def test_merge_samples_collapses_to_groups(ps: Phyloseq) -> None:
    ps2 = merge_samples(ps, "Group")
    assert ps2.nsamples == 2


def test_merge_samples_sums_otu_abundances(ps: Phyloseq) -> None:
    orig = ps.otu_table.to_dataframe()
    ps2 = merge_samples(ps, "Group")
    result = ps2.otu_table.to_dataframe()
    for grp, samples in [("A", ["S1", "S2"]), ("B", ["S3", "S4"])]:
        if grp in result.columns:
            np.testing.assert_allclose(
                result[grp].values,
                orig[samples].sum(axis=1).values,
                atol=1e-10,
            )


def test_merge_samples_no_sample_data_raises() -> None:
    ps = _make_ps(with_sam=False)
    with pytest.raises(pyloseq.pyloseqValidationError):
        merge_samples(ps, "Group")


def test_merge_samples_bad_variable_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        merge_samples(ps, "NoSuchColumn")


def test_merge_samples_custom_fn_sums_instead_of_mean(ps: Phyloseq) -> None:
    orig = ps.otu_table.to_dataframe()
    ps2 = merge_samples(ps, "Group", fn=np.sum)
    result = ps2.otu_table.to_dataframe()
    for grp, samples in [("A", ["S1", "S2"]), ("B", ["S3", "S4"])]:
        if grp in result.columns:
            np.testing.assert_allclose(
                result[grp].values,
                orig[samples].sum(axis=1).values,
                atol=1e-10,
            )


def test_merge_samples_preserves_refseq() -> None:
    df = pd.DataFrame({"S1": [5.0, 3.0], "S2": [2.0, 1.0]}, index=["OTU1", "OTU2"])
    sam_df = pd.DataFrame({"Group": ["A", "A"]}, index=["S1", "S2"])
    rs = RefSeq({"OTU1": skbio.DNA("ACGT"), "OTU2": skbio.DNA("TTTT")})
    ps = Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True), sam=SampleData(sam_df), refseq=rs
    )
    ps2 = merge_samples(ps, "Group")
    assert ps2.refseq is not None


@pytest.mark.skipif(not GP_MERGESAM_PRESENT, reason="golden files not generated yet")
def test_merge_samples_sampletype_matches_r() -> None:
    ref = load_global_patterns_reference()
    gp = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
    )
    gp2 = merge_samples(gp, "SampleType")

    golden_ss = pd.read_parquet(
        GP_GOLDEN / "merge_samples_SampleType" / "sample_sums.parquet"
    )
    if "__index__" in golden_ss.columns:
        golden_ss = golden_ss.set_index("__index__")
        golden_ss.index.name = None

    assert gp2.nsamples == len(golden_ss)
    np.testing.assert_allclose(
        gp2.sample_sums().sum(),
        golden_ss["value"].sum(),
        atol=1e-6,
    )


# ===========================================================================
# psmelt
# ===========================================================================


def test_psmelt_shape(ps: Phyloseq) -> None:
    long = psmelt(ps)
    assert len(long) == ps.ntaxa * ps.nsamples


def test_psmelt_required_columns(ps: Phyloseq) -> None:
    long = psmelt(ps)
    assert "OTU" in long.columns
    assert "Sample" in long.columns
    assert "Abundance" in long.columns


def test_psmelt_sample_variables_present(ps: Phyloseq) -> None:
    long = psmelt(ps)
    for v in ps.sample_variables:
        assert v in long.columns


def test_psmelt_rank_names_present(ps: Phyloseq) -> None:
    long = psmelt(ps)
    for r in ps.rank_names:
        assert r in long.columns


def test_psmelt_abundance_sum(ps: Phyloseq) -> None:
    long = psmelt(ps)
    np.testing.assert_allclose(
        long["Abundance"].sum(),
        ps.otu_table.to_dataframe().values.sum(),
        atol=1e-10,
    )


def test_psmelt_melt_method_alias(ps: Phyloseq) -> None:
    assert ps.melt().equals(psmelt(ps))


def test_psmelt_no_sample_data() -> None:
    ps = _make_ps(with_sam=False, with_tax=False)
    long = psmelt(ps)
    assert len(long) == ps.ntaxa * ps.nsamples
    assert "OTU" in long.columns
