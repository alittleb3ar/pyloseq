"""Phase 3 — Data manipulation tests.

All tests use programmatic fixtures built in tmp_path; golden-file
comparisons are skipped when the golden directory is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import phyla
from phyla import (
    OtuTable,
    Phyloseq,
    SampleData,
    TaxTable,
    filter_taxa,
    kOverA,
    merge_phyloseq,
    merge_samples,
    merge_taxa,
    prune_samples,
    prune_taxa,
    psmelt,
    rarefy_even_depth,
    subset_samples,
    subset_taxa,
    tax_glom,
    tip_glom,
    transform_sample_counts,
)

GOLDEN_DIR = Path("tests/golden")
GP_GOLDEN = GOLDEN_DIR / "GlobalPatterns"
ET_GOLDEN = GOLDEN_DIR / "enterotype"

GOLDEN_PRESENT = (GP_GOLDEN / "otu_table.parquet").exists()
GP_SUBSET_SOIL_PRESENT = (GP_GOLDEN / "subset_samples_soil" / "otu_table.parquet").exists()
GP_SUBSET_CHLAM_PRESENT = (GP_GOLDEN / "subset_taxa_chlamydiae" / "otu_table.parquet").exists()
ET_FILTER_PRESENT = (ET_GOLDEN / "filter_taxa_kOverA_5_2e-5" / "otu_table.parquet").exists()
GP_TAXGLOM_PRESENT = (GP_GOLDEN / "tax_glom_Family" / "taxa_sums.parquet").exists()
GP_MERGESAM_PRESENT = (GP_GOLDEN / "merge_samples_SampleType" / "sample_sums.parquet").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ps(
    ntaxa: int = 6,
    nsamples: int = 4,
    with_sam: bool = True,
    with_tax: bool = True,
    with_tree: bool = False,
    rng: np.random.Generator | None = None,
) -> Phyloseq:
    if rng is None:
        rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(ntaxa, nsamples)).astype(float)
    taxa = [f"OTU{i+1}" for i in range(ntaxa)]
    samples = [f"S{i+1}" for i in range(nsamples)]
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
        phyla_vals = ["Firmicutes", "Firmicutes", "Bacteroidetes", "Proteobacteria",
                      "Proteobacteria", "Chlamydiae"][:ntaxa]
        genus_vals = ["Genus_A", "Genus_A", "Genus_B", "Genus_C", "Genus_D", "Genus_E"][:ntaxa]
        tax_df = pd.DataFrame(
            {"Phylum": phyla_vals, "Genus": genus_vals},
            index=taxa,
        )
        tax = TaxTable(tax_df)

    return Phyloseq(otu=otu, sam=sam, tax=tax)


def _make_ps_with_tree() -> Phyloseq:
    """Small 4-taxa PS with a star tree."""
    from io import StringIO

    import skbio.tree

    from phyla import PhyTree

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


# ---------------------------------------------------------------------------
# Ticket 3.2 — prune_taxa, prune_samples
# ---------------------------------------------------------------------------


class TestPruning:
    def test_prune_taxa_basic(self) -> None:
        ps = _make_ps()
        ps2 = prune_taxa(["OTU1", "OTU3"], ps)
        assert list(ps2.taxa_names) == ["OTU1", "OTU3"]
        assert ps2.ntaxa == 2
        assert ps2.nsamples == ps.nsamples

    def test_prune_taxa_order_preserved(self) -> None:
        ps = _make_ps()
        ps2 = prune_taxa(["OTU4", "OTU1", "OTU2"], ps)
        assert list(ps2.taxa_names) == ["OTU4", "OTU1", "OTU2"]

    def test_prune_taxa_absent_names_ignored(self) -> None:
        ps = _make_ps()
        ps2 = prune_taxa(["OTU1", "UNKNOWN", "OTU2"], ps)
        assert list(ps2.taxa_names) == ["OTU1", "OTU2"]

    def test_prune_taxa_no_mutation(self) -> None:
        ps = _make_ps()
        original_ntaxa = ps.ntaxa
        _ = prune_taxa(["OTU1"], ps)
        assert ps.ntaxa == original_ntaxa

    def test_prune_samples_basic(self) -> None:
        ps = _make_ps()
        ps2 = prune_samples(["S1", "S3"], ps)
        assert list(ps2.sample_names) == ["S1", "S3"]
        assert ps2.nsamples == 2
        assert ps2.ntaxa == ps.ntaxa

    def test_prune_samples_updates_sample_data(self) -> None:
        ps = _make_ps()
        ps2 = prune_samples(["S2"], ps)
        assert ps2.sample_data is not None
        assert list(ps2.sample_names) == ["S2"]

    def test_prune_samples_no_mutation(self) -> None:
        ps = _make_ps()
        original_nsamples = ps.nsamples
        _ = prune_samples(["S1"], ps)
        assert ps.nsamples == original_nsamples


# ---------------------------------------------------------------------------
# Ticket 3.1 — subset_samples, subset_taxa
# ---------------------------------------------------------------------------


class TestSubset:
    def test_subset_samples_lambda(self) -> None:
        ps = _make_ps()
        ps2 = subset_samples(ps, lambda s: s["Group"] == "A")
        assert all(ps2.sample_data.to_frame()["Group"] == "A")

    def test_subset_samples_query_string(self) -> None:
        ps = _make_ps()
        ps2 = subset_samples(ps, 'Group == "B"')
        assert all(ps2.sample_data.to_frame()["Group"] == "B")

    def test_subset_samples_preserves_taxa(self) -> None:
        ps = _make_ps()
        ps2 = subset_samples(ps, lambda s: s["Group"] == "A")
        assert ps2.ntaxa == ps.ntaxa

    def test_subset_taxa_lambda(self) -> None:
        ps = _make_ps()
        ps2 = subset_taxa(ps, lambda t: t["Phylum"] == "Firmicutes")
        assert ps2.ntaxa == 2  # OTU1 and OTU2 are Firmicutes
        assert all(ps2.tax_table.to_frame()["Phylum"] == "Firmicutes")

    def test_subset_taxa_query_string(self) -> None:
        ps = _make_ps()
        ps2 = subset_taxa(ps, 'Phylum == "Chlamydiae"')
        assert ps2.ntaxa == 1

    def test_subset_samples_no_sample_data_raises(self) -> None:
        ps = _make_ps(with_sam=False)
        with pytest.raises(phyla.PhylaValidationError):
            subset_samples(ps, lambda s: True)

    def test_subset_taxa_no_tax_table_raises(self) -> None:
        ps = _make_ps(with_tax=False)
        with pytest.raises(phyla.PhylaValidationError):
            subset_taxa(ps, lambda t: True)

    @pytest.mark.skipif(not GP_SUBSET_SOIL_PRESENT, reason="golden files not generated yet")
    def test_subset_samples_soil_matches_r(self) -> None:
        """subset_samples(GP, SampleType == 'Soil') → 3 samples."""
        from phyla.testing.fixtures import load_global_patterns_reference

        ref = load_global_patterns_reference()
        gp = Phyloseq(
            otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
            sam=SampleData(ref["sample_data"]),
            tax=TaxTable(ref["tax_table"]),
        )
        gp_soil = subset_samples(gp, 'SampleType == "Soil"')
        assert gp_soil.nsamples == 3

        # Compare against golden
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

    @pytest.mark.skipif(not GP_SUBSET_CHLAM_PRESENT, reason="golden files not generated yet")
    def test_subset_taxa_chlamydiae_matches_r(self) -> None:
        from phyla.testing.fixtures import load_global_patterns_reference

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


# ---------------------------------------------------------------------------
# Ticket 3.3 — filter_taxa, kOverA
# ---------------------------------------------------------------------------


class TestFilterTaxa:
    def test_kOverA_basic(self) -> None:
        pred = kOverA(2, 10.0)
        assert pred(pd.Series([0.0, 11.0, 12.0, 5.0])) is True
        assert pred(pd.Series([0.0, 11.0, 5.0, 5.0])) is False

    def test_filter_taxa_prune_true(self) -> None:
        ps = _make_ps(rng=np.random.default_rng(1))
        ps2 = filter_taxa(ps, kOverA(1, 0.0), prune=True)
        assert isinstance(ps2, Phyloseq)
        assert ps2.ntaxa <= ps.ntaxa

    def test_filter_taxa_prune_false_returns_series(self) -> None:
        ps = _make_ps()
        mask = filter_taxa(ps, kOverA(1, 0.0), prune=False)
        assert isinstance(mask, pd.Series)
        assert mask.index.equals(ps.taxa_names)

    def test_filter_taxa_all_pass(self) -> None:
        ps = _make_ps()
        ps2 = filter_taxa(ps, lambda x: True, prune=True)
        assert ps2.ntaxa == ps.ntaxa

    def test_filter_taxa_none_pass(self) -> None:
        ps = _make_ps()
        ps2 = filter_taxa(ps, lambda x: False, prune=True)
        assert ps2.ntaxa == 0

    @pytest.mark.skipif(not ET_FILTER_PRESENT, reason="golden files not generated yet")
    def test_filter_taxa_kOverA_matches_r_enterotype(self) -> None:
        """filter_taxa(enterotype, kOverA(5, 2e-5)) → 416 taxa × 280 samples."""
        from phyla.testing.fixtures import load_enterotype_reference

        ref = load_enterotype_reference()
        et = Phyloseq(
            otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
            sam=SampleData(ref["sample_data"]),
        )
        et2 = filter_taxa(et, kOverA(5, 2e-5), prune=True)

        golden_otu = pd.read_parquet(
            ET_GOLDEN / "filter_taxa_kOverA_5_2e-5" / "otu_table.parquet"
        )
        if "__index__" in golden_otu.columns:
            golden_otu = golden_otu.set_index("__index__")
            golden_otu.index.name = None
        assert et2.ntaxa == len(golden_otu)
        assert et2.nsamples == 280


# ---------------------------------------------------------------------------
# Ticket 3.4 — transform_sample_counts
# ---------------------------------------------------------------------------


class TestTransform:
    def test_relative_abundance(self) -> None:
        ps = _make_ps()
        ps2 = transform_sample_counts(ps, lambda x: x / x.sum())
        col_sums = ps2.otu_table.to_dataframe().sum(axis=0)
        np.testing.assert_allclose(col_sums.values, 1.0, atol=1e-12)

    def test_all_zero_sample_produces_nan(self) -> None:
        df = pd.DataFrame(
            {"S1": [0.0, 0.0], "S2": [5.0, 3.0]},
            index=["OTU1", "OTU2"],
        )
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
        ps2 = transform_sample_counts(ps, lambda x: x / x.sum())
        assert ps2.otu_table.to_dataframe()["S1"].isna().all()

    def test_log_transform(self) -> None:
        ps = _make_ps()
        ps2 = transform_sample_counts(ps, lambda x: np.log1p(x))
        orig = ps.otu_table.to_dataframe()
        result = ps2.otu_table.to_dataframe()
        np.testing.assert_allclose(result.values, np.log1p(orig.values), atol=1e-14)

    def test_no_mutation(self) -> None:
        ps = _make_ps()
        orig_sum = ps.taxa_sums().sum()
        _ = transform_sample_counts(ps, lambda x: x / x.sum())
        assert ps.taxa_sums().sum() == orig_sum


# ---------------------------------------------------------------------------
# Ticket 3.5 — rarefy_even_depth
# ---------------------------------------------------------------------------


class TestRarefy:
    def test_all_samples_equal_depth(self) -> None:
        ps = _make_ps(rng=np.random.default_rng(99))
        depth = 20
        ps2 = rarefy_even_depth(ps, sample_size=depth, verbose=False)
        ss = ps2.sample_sums()
        np.testing.assert_allclose(ss.values, depth, atol=0)

    def test_low_samples_dropped(self) -> None:
        df = pd.DataFrame(
            {"S1": [5.0, 3.0], "S2": [100.0, 200.0]},
            index=["OTU1", "OTU2"],
        )
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
        with pytest.warns(UserWarning, match="Dropping"):
            ps2 = rarefy_even_depth(ps, sample_size=50, verbose=True)
        assert "S1" not in list(ps2.sample_names)
        assert "S2" in list(ps2.sample_names)

    def test_reproducibility(self) -> None:
        ps = _make_ps(rng=np.random.default_rng(7))
        ps_a = rarefy_even_depth(ps, sample_size=15, rng_seed=42, verbose=False)
        ps_b = rarefy_even_depth(ps, sample_size=15, rng_seed=42, verbose=False)
        np.testing.assert_array_equal(
            ps_a.otu_table.to_dataframe().values,
            ps_b.otu_table.to_dataframe().values,
        )

    def test_trim_otus(self) -> None:
        df = pd.DataFrame(
            {"S1": [100.0, 0.0, 50.0], "S2": [200.0, 0.0, 30.0]},
            index=["OTU1", "OTU2", "OTU3"],
        )
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
        ps2 = rarefy_even_depth(ps, sample_size=20, trim_otus=True, verbose=False)
        assert "OTU2" not in list(ps2.taxa_names)

    def test_replace_false(self) -> None:
        df = pd.DataFrame(
            {"S1": [50.0, 30.0, 20.0], "S2": [60.0, 20.0, 20.0]},
            index=["OTU1", "OTU2", "OTU3"],
        )
        ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True))
        ps2 = rarefy_even_depth(ps, sample_size=10, replace=False, verbose=False)
        ss = ps2.sample_sums()
        np.testing.assert_allclose(ss.values, 10.0, atol=0)

    def test_no_mutation(self) -> None:
        ps = _make_ps(rng=np.random.default_rng(3))
        orig = ps.taxa_sums().sum()
        _ = rarefy_even_depth(ps, sample_size=10, verbose=False)
        assert ps.taxa_sums().sum() == orig


# ---------------------------------------------------------------------------
# Ticket 3.6 — tax_glom
# ---------------------------------------------------------------------------


class TestTaxGlom:
    def test_glom_collapses_taxa(self) -> None:
        ps = _make_ps()
        # Phylum has 4 distinct values: Firmicutes(2), Bacteroidetes(1),
        # Proteobacteria(2), Chlamydiae(1) → 4 groups
        ps2 = tax_glom(ps, "Phylum")
        assert ps2.ntaxa == 4

    def test_glom_sums_abundances(self) -> None:
        ps = _make_ps()
        ps2 = tax_glom(ps, "Phylum")
        # Total abundance must be preserved (na_rm may drop some)
        orig_total = ps.taxa_sums().sum()
        # All taxa have non-NA Phylum so total should equal
        assert abs(ps2.taxa_sums().sum() - orig_total) < 1e-10

    def test_glom_at_genus_gives_more_groups(self) -> None:
        ps = _make_ps()
        ps_phylum = tax_glom(ps, "Phylum")
        ps_genus = tax_glom(ps, "Genus")
        assert ps_genus.ntaxa >= ps_phylum.ntaxa

    def test_glom_no_tax_raises(self) -> None:
        ps = _make_ps(with_tax=False)
        with pytest.raises(phyla.PhylaValidationError):
            tax_glom(ps, "Phylum")

    def test_glom_bad_rank_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
            tax_glom(ps, "Species")

    def test_glom_preserves_sample_data(self) -> None:
        ps = _make_ps()
        ps2 = tax_glom(ps, "Phylum")
        assert ps2.sample_data is not None
        assert ps2.nsamples == ps.nsamples

    def test_glom_na_rm(self) -> None:
        ps = _make_ps()
        tax_df = ps.tax_table.to_frame().copy()
        tax_df.loc["OTU1", "Phylum"] = ""
        ps2 = Phyloseq(
            otu=ps.otu_table.copy(),
            tax=TaxTable(tax_df),
        )
        ps3 = tax_glom(ps2, "Phylum", na_rm=True)
        # OTU1's Phylum="" should be dropped
        assert ps3.ntaxa < ps.ntaxa

    @pytest.mark.skipif(not GP_TAXGLOM_PRESENT, reason="golden files not generated yet")
    def test_tax_glom_family_matches_r(self) -> None:
        from phyla.testing.fixtures import load_global_patterns_reference

        ref = load_global_patterns_reference()
        gp = Phyloseq(
            otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
            tax=TaxTable(ref["tax_table"]),
        )
        gp_fam = tax_glom(gp, "Family")

        golden_ts = pd.read_parquet(
            GP_GOLDEN / "tax_glom_Family" / "taxa_sums.parquet"
        )
        if "__index__" in golden_ts.columns:
            golden_ts = golden_ts.set_index("__index__")
            golden_ts.index.name = None

        assert gp_fam.ntaxa == len(golden_ts)
        # Total abundance must match (all taxa_sums sum to same value)
        np.testing.assert_allclose(
            gp_fam.taxa_sums().sum(),
            golden_ts["value"].sum(),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Ticket 3.7 — tip_glom
# ---------------------------------------------------------------------------


class TestTipGlom:
    def test_tip_glom_reduces_taxa(self) -> None:
        ps = _make_ps_with_tree()
        # Tree has OTU1+OTU2 close (dist=0.2), OTU3+OTU4 close (dist=0.3)
        ps2 = tip_glom(ps, h=0.25)
        assert ps2.ntaxa <= ps.ntaxa

    def test_tip_glom_no_tree_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
            tip_glom(ps, h=0.1)

    def test_tip_glom_large_h_merges_all(self) -> None:
        ps = _make_ps_with_tree()
        ps2 = tip_glom(ps, h=100.0)
        assert ps2.ntaxa == 1

    def test_tip_glom_zero_h_keeps_all(self) -> None:
        ps = _make_ps_with_tree()
        # h=0 means no two tips are within distance 0, so no merging
        ps2 = tip_glom(ps, h=0.0)
        assert ps2.ntaxa == ps.ntaxa


# ---------------------------------------------------------------------------
# Ticket 3.8 — merge_taxa, merge_phyloseq, merge_samples
# ---------------------------------------------------------------------------


class TestMergeTaxa:
    def test_merge_reduces_ntaxa(self) -> None:
        ps = _make_ps()
        ps2 = merge_taxa(ps, ["OTU1", "OTU2"])
        assert ps2.ntaxa == ps.ntaxa - 1

    def test_merge_sums_abundances(self) -> None:
        ps = _make_ps()
        orig = ps.otu_table.to_dataframe()
        ps2 = merge_taxa(ps, ["OTU1", "OTU2"])
        result = ps2.otu_table.to_dataframe()
        archetype = (result.index.intersection(pd.Index(["OTU1", "OTU2"])))[0]
        np.testing.assert_allclose(
            result.loc[archetype].values,
            (orig.loc["OTU1"] + orig.loc["OTU2"]).values,
            atol=1e-10,
        )

    def test_merge_explicit_archetype(self) -> None:
        ps = _make_ps()
        ps2 = merge_taxa(ps, ["OTU1", "OTU2", "OTU3"], archetype="OTU2")
        assert "OTU2" in list(ps2.taxa_names)
        assert "OTU1" not in list(ps2.taxa_names)

    def test_merge_single_taxon_noop(self) -> None:
        ps = _make_ps()
        ps2 = merge_taxa(ps, ["OTU1"])
        assert ps2.ntaxa == ps.ntaxa

    def test_merge_total_abundance_preserved(self) -> None:
        ps = _make_ps()
        orig_total = ps.taxa_sums().sum()
        ps2 = merge_taxa(ps, ["OTU1", "OTU2", "OTU3"])
        assert abs(ps2.taxa_sums().sum() - orig_total) < 1e-10


class TestMergePhyloseq:
    def test_merge_two_disjoint_samples(self) -> None:
        ps1 = _make_ps(nsamples=2, with_sam=False, with_tax=False)
        # Create a second with different sample names
        df2 = pd.DataFrame(
            {"S5": [1.0, 2.0], "S6": [3.0, 4.0]},
            index=["OTU1", "OTU2"],
        )
        ps2 = Phyloseq(otu=OtuTable(df2, taxa_are_rows=True))
        merged = merge_phyloseq(ps1, ps2)
        assert merged.nsamples == ps1.nsamples + ps2.nsamples

    def test_merge_sums_overlapping_abundances(self) -> None:
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

    def test_merge_union_taxa(self) -> None:
        df1 = pd.DataFrame({"S1": [10.0]}, index=["OTU1"])
        df2 = pd.DataFrame({"S2": [5.0]}, index=["OTU2"])
        ps1 = Phyloseq(otu=OtuTable(df1, taxa_are_rows=True))
        ps2 = Phyloseq(otu=OtuTable(df2, taxa_are_rows=True))
        merged = merge_phyloseq(ps1, ps2)
        assert merged.ntaxa == 2
        assert merged.nsamples == 2

    def test_merge_requires_two_or_more(self) -> None:
        ps = _make_ps()
        with pytest.raises(ValueError):
            merge_phyloseq(ps)


class TestMergeSamples:
    def test_merge_collapses_to_groups(self) -> None:
        ps = _make_ps()
        ps2 = merge_samples(ps, "Group")
        assert ps2.nsamples == 2  # Groups A and B

    def test_merge_sums_otu_abundances(self) -> None:
        ps = _make_ps()
        orig = ps.otu_table.to_dataframe()
        ps2 = merge_samples(ps, "Group")
        result = ps2.otu_table.to_dataframe()
        # Group A = S1 + S2; Group B = S3 + S4
        for grp, samples in [("A", ["S1", "S2"]), ("B", ["S3", "S4"])]:
            if grp in result.columns:
                np.testing.assert_allclose(
                    result[grp].values,
                    orig[samples].sum(axis=1).values,
                    atol=1e-10,
                )

    def test_merge_no_sample_data_raises(self) -> None:
        ps = _make_ps(with_sam=False)
        with pytest.raises(phyla.PhylaValidationError):
            merge_samples(ps, "Group")

    def test_merge_bad_variable_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
            merge_samples(ps, "NoSuchColumn")

    @pytest.mark.skipif(not GP_MERGESAM_PRESENT, reason="golden files not generated yet")
    def test_merge_samples_sampletype_matches_r(self) -> None:
        """merge_samples(GP, 'SampleType') → 9 samples (one per SampleType)."""
        from phyla.testing.fixtures import load_global_patterns_reference

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
        # Total abundance must be conserved (summing doesn't lose reads)
        np.testing.assert_allclose(
            gp2.sample_sums().sum(),
            golden_ss["value"].sum(),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Ticket 3.9 — psmelt
# ---------------------------------------------------------------------------


class TestPsmelt:
    def test_shape(self) -> None:
        ps = _make_ps()
        long = psmelt(ps)
        assert len(long) == ps.ntaxa * ps.nsamples

    def test_required_columns(self) -> None:
        ps = _make_ps()
        long = psmelt(ps)
        assert "OTU" in long.columns
        assert "Sample" in long.columns
        assert "Abundance" in long.columns

    def test_sample_variables_present(self) -> None:
        ps = _make_ps()
        long = psmelt(ps)
        for v in ps.sample_variables:
            assert v in long.columns

    def test_rank_names_present(self) -> None:
        ps = _make_ps()
        long = psmelt(ps)
        for r in ps.rank_names:
            assert r in long.columns

    def test_abundance_sum(self) -> None:
        ps = _make_ps()
        long = psmelt(ps)
        np.testing.assert_allclose(
            long["Abundance"].sum(),
            ps.otu_table.to_dataframe().values.sum(),
            atol=1e-10,
        )

    def test_melt_method_alias(self) -> None:
        ps = _make_ps()
        assert ps.melt().equals(psmelt(ps))

    def test_no_sample_data(self) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        long = psmelt(ps)
        assert len(long) == ps.ntaxa * ps.nsamples
        assert "OTU" in long.columns


# ---------------------------------------------------------------------------
# API surface check
# ---------------------------------------------------------------------------


def test_manipulation_functions_exported() -> None:
    for name in [
        "subset_samples", "subset_taxa",
        "prune_samples", "prune_taxa",
        "filter_taxa", "kOverA",
        "transform_sample_counts",
        "rarefy_even_depth",
        "tax_glom", "tip_glom",
        "merge_phyloseq", "merge_samples", "merge_taxa",
        "psmelt",
    ]:
        assert hasattr(phyla, name), f"phyla.{name} not exported"
