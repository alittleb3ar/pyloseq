"""Phase 4 — Analysis tests.

Golden-file comparisons are skipped when the relevant golden files are absent.
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
    distance,
    distance_method_list,
    estimate_richness,
    ordinate,
    unifrac,
)

GOLDEN_DIR = Path("tests/golden")
GP_GOLDEN = GOLDEN_DIR / "GlobalPatterns"
ES_GOLDEN = GOLDEN_DIR / "esophagus"

GP_PRESENT = (GP_GOLDEN / "otu_table.parquet").exists()
ES_PRESENT = (ES_GOLDEN / "otu_table.parquet").exists()
RICH_PRESENT = (GP_GOLDEN / "estimate_richness" / "default.parquet").exists()
UF_UN_PRESENT = (ES_GOLDEN / "unifrac_unweighted" / "normalized.parquet").exists()
UF_WT_PRESENT = (ES_GOLDEN / "unifrac_weighted" / "normalized.parquet").exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ps(ntaxa: int = 6, nsamples: int = 4) -> Phyloseq:
    rng = np.random.default_rng(42)
    counts = rng.integers(1, 100, size=(ntaxa, nsamples)).astype(float)
    taxa = [f"OTU{i+1}" for i in range(ntaxa)]
    samples = [f"S{i+1}" for i in range(nsamples)]
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

    from phyla import PhyTree

    newick = "((OTU1:0.1,OTU2:0.2):0.1,(OTU3:0.15,OTU4:0.05):0.2);"
    tree_node = skbio.tree.TreeNode.read(
        StringIO(newick), format="newick", convert_underscores=False
    )
    rng = np.random.default_rng(7)
    counts = rng.integers(1, 200, size=(4, 5)).astype(float)
    taxa = ["OTU1", "OTU2", "OTU3", "OTU4"]
    samples = [f"S{i+1}" for i in range(5)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        tree=PhyTree(tree_node),
    )


def _load_esophagus() -> Phyloseq:
    from phyla import PhyTree
    from phyla.testing.fixtures import load_esophagus_reference

    ref = load_esophagus_reference()
    tree = PhyTree.from_newick(ref["phy_tree_newick"]) if "phy_tree_newick" in ref else None
    return Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tree=tree,
    )


# ---------------------------------------------------------------------------
# Ticket 4.1 — estimate_richness
# ---------------------------------------------------------------------------


class TestEstimateRichness:
    def test_returns_dataframe(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == ps.nsamples

    def test_default_measures(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps)
        expected = ["Observed", "Chao1", "se.chao1", "ACE", "se.ACE",
                    "Shannon", "Simpson", "InvSimpson", "Fisher"]
        assert list(df.columns) == expected

    def test_subset_measures(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps, measures=["Observed", "Shannon"])
        assert list(df.columns) == ["Observed", "Shannon"]

    def test_observed_is_nonzero_taxa(self) -> None:
        df_data = pd.DataFrame(
            {"S1": [10.0, 0.0, 5.0], "S2": [0.0, 3.0, 7.0]},
            index=["OTU1", "OTU2", "OTU3"],
        )
        ps = Phyloseq(otu=OtuTable(df_data, taxa_are_rows=True))
        df = estimate_richness(ps, measures=["Observed"])
        assert df.loc["S1", "Observed"] == 2.0
        assert df.loc["S2", "Observed"] == 2.0

    def test_shannon_relative_entropy(self) -> None:
        # Uniform distribution of 4 taxa → Shannon = ln(4)
        df_data = pd.DataFrame(
            {"S1": [25.0, 25.0, 25.0, 25.0]},
            index=["OTU1", "OTU2", "OTU3", "OTU4"],
        )
        ps = Phyloseq(otu=OtuTable(df_data, taxa_are_rows=True))
        df = estimate_richness(ps, measures=["Shannon"])
        np.testing.assert_allclose(df.loc["S1", "Shannon"], np.log(4), atol=1e-12)

    def test_simpson_range(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps, measures=["Simpson"])
        assert ((df["Simpson"] >= 0) & (df["Simpson"] <= 1)).all()

    def test_invsimpson_reciprocal(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps, measures=["Simpson", "InvSimpson"])
        # InvSimpson = 1 / (1 - Simpson) = 1 / D
        # Simpson = 1 - D  →  D = 1 - Simpson  →  InvSimpson = 1/(1-Simpson)
        d = 1.0 - df["Simpson"].values
        expected_inv = 1.0 / d
        np.testing.assert_allclose(df["InvSimpson"].values, expected_inv, atol=1e-12)

    def test_fisher_positive(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps, measures=["Fisher"])
        assert (df["Fisher"] > 0).all()

    def test_chao1_ge_observed(self) -> None:
        ps = _make_ps()
        df = estimate_richness(ps, measures=["Observed", "Chao1"])
        assert (df["Chao1"] >= df["Observed"]).all()

    def test_bad_measure_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
            estimate_richness(ps, measures=["NotAMeasure"])

    @pytest.mark.skipif(not RICH_PRESENT, reason="golden files not generated yet")
    def test_estimate_richness_matches_r_globalpatterns(self) -> None:
        from phyla.testing.fixtures import load_global_patterns_reference

        ref = load_global_patterns_reference()
        gp = Phyloseq(otu=OtuTable(ref["otu_table"], taxa_are_rows=True))
        result = estimate_richness(gp)

        golden = pd.read_parquet(GP_GOLDEN / "estimate_richness" / "default.parquet")
        if "__index__" in golden.columns:
            golden = golden.set_index("__index__")
            golden.index.name = None

        common_samples = result.index.intersection(golden.index)
        for measure in ["Observed", "Shannon", "Simpson", "InvSimpson"]:
            if measure in golden.columns:
                np.testing.assert_allclose(
                    result.loc[common_samples, measure].values,
                    golden.loc[common_samples, measure].values,
                    atol=1e-9,
                    err_msg=f"Mismatch for measure: {measure}",
                )

        # Fisher's alpha: match to 1e-4 (numerical solve vs R's uniroot)
        if "Fisher" in golden.columns:
            np.testing.assert_allclose(
                result.loc[common_samples, "Fisher"].values,
                golden.loc[common_samples, "Fisher"].values,
                atol=1e-4,
            )


# ---------------------------------------------------------------------------
# Ticket 4.2 — distance dispatcher
# ---------------------------------------------------------------------------


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
        """phyloseq Jaccard uses binary transform."""
        ps = _make_ps()
        dm = distance(ps, "jaccard")
        arr = np.array(dm.data)
        # All distances should be in [0, 1]
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0 + 1e-12

    def test_jsd(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps()
        dm = distance(ps, "jsd")
        assert isinstance(dm, DistanceMatrix)

    def test_unknown_method_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
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


# ---------------------------------------------------------------------------
# Ticket 4.3 — UniFrac
# ---------------------------------------------------------------------------


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
        with pytest.raises(phyla.PhylaValidationError):
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
        np.testing.assert_allclose(
            np.array(dm1.data), np.array(dm2.data), atol=1e-12
        )

    @pytest.mark.skipif(not (ES_PRESENT and UF_UN_PRESENT), reason="golden files not generated yet")
    def test_unweighted_unifrac_matches_r_esophagus(self) -> None:
        """Unweighted UniFrac C-B ≈ 0.5176, D-B ≈ 0.5182, D-C ≈ 0.5422 (atol=1e-4)."""
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
        """Weighted UniFrac C-B ≈ 0.2035, D-B ≈ 0.2603, D-C ≈ 0.2477 (atol=1e-4)."""
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


# ---------------------------------------------------------------------------
# Ticket 4.5 — ordinate
# ---------------------------------------------------------------------------


class TestOrdinate:
    def test_pcoa_returns_ordination(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ordinate(ps, method="PCoA", distance="bray")
        assert isinstance(result, OrdinationResults)

    def test_mds_alias(self) -> None:
        ps = _make_ps()
        r1 = ordinate(ps, method="PCoA", distance="bray")
        r2 = ordinate(ps, method="MDS", distance="bray")
        np.testing.assert_allclose(
            np.abs(r1.samples.values), np.abs(r2.samples.values), atol=1e-10
        )

    def test_pcoa_sample_count(self) -> None:
        ps = _make_ps()
        result = ordinate(ps, method="PCoA", distance="euclidean")
        assert len(result.samples) == ps.nsamples

    def test_pcoa_proportion_explained_sums_to_one(self) -> None:
        ps = _make_ps()
        result = ordinate(ps, method="PCoA", distance="euclidean")
        if result.proportion_explained is not None:
            total = result.proportion_explained.dropna().sum()
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_nmds_returns_ordination(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ordinate(ps, method="NMDS", distance="bray")
        assert isinstance(result, OrdinationResults)
        assert len(result.samples) == ps.nsamples

    def test_unknown_method_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
            ordinate(ps, method="GARBAGE")

    def test_dca_not_implemented(self) -> None:
        ps = _make_ps()
        with pytest.raises(NotImplementedError):
            ordinate(ps, method="DCA")

    def test_ordinate_with_precomputed_dm(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "bray")
        result = ordinate(ps, method="PCoA", distance=dm)
        assert len(result.samples) == ps.nsamples

    def test_cca_requires_formula(self) -> None:
        ps = _make_ps()
        with pytest.raises(phyla.PhylaValidationError):
            ordinate(ps, method="CCA", formula=None)

    def test_rda_with_formula(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ordinate(ps, method="RDA", formula="~Group")
        assert isinstance(result, OrdinationResults)


# ---------------------------------------------------------------------------
# Ticket 4.6 — plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    def test_plot_bar_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = phyla.plot_bar(ps)
        assert isinstance(p, ggplot)

    def test_plot_bar_with_fill(self) -> None:
        from plotnine import ggplot


        ps = _make_ps()
        tax_df = pd.DataFrame(
            {"Phylum": ["A", "A", "B", "B", "C", "C"]},
            index=[f"OTU{i+1}" for i in range(6)],
        )
        ps2 = Phyloseq(otu=ps.otu_table.copy(), tax=TaxTable(tax_df))
        p = phyla.plot_bar(ps2, fill="Phylum")
        assert isinstance(p, ggplot)

    def test_plot_richness_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = phyla.plot_richness(ps, measures=["Observed", "Shannon"])
        assert isinstance(p, ggplot)

    def test_plot_ordination_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        ord_result = ordinate(ps, method="PCoA", distance="bray")
        p = phyla.plot_ordination(ps, ord_result, type="samples")
        assert isinstance(p, ggplot)

    def test_plot_ordination_scree(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        ord_result = ordinate(ps, method="PCoA", distance="euclidean")
        p = phyla.plot_ordination(ps, ord_result, type="scree")
        assert isinstance(p, ggplot)

    def test_plot_heatmap_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = phyla.plot_heatmap(ps, method="PCoA", distance="bray")
        assert isinstance(p, ggplot)

    def test_plot_heatmap_log4_trans(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = phyla.plot_heatmap(ps, trans="log4")
        assert isinstance(p, ggplot)


# ---------------------------------------------------------------------------
# API surface check
# ---------------------------------------------------------------------------


def test_phase4_functions_exported() -> None:
    for name in [
        "estimate_richness",
        "distance", "distance_method_list", "unifrac",
        "ordinate",
        "plot_bar", "plot_richness", "plot_ordination", "plot_heatmap",
        "make_network", "plot_network",
    ]:
        assert hasattr(phyla, name), f"phyla.{name} not exported"
