"""Tests for pyloseq plotting functions.

All tests are smoke / structural — they verify return types, required columns,
and edge-case logic without rendering to a canvas.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from plotnine.scales.scale_xy import scale_x_discrete as ScaleXDiscrete
from plotnine.scales.scale_xy import scale_y_discrete as ScaleYDiscrete
from skbio.stats.ordination import OrdinationResults

from pyloseq import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable
from pyloseq._diversity import _ALL_MEASURES
from pyloseq._exceptions import pyloseqValidationError
from pyloseq._ordination import ordinate
from pyloseq.plotting import (
    _convex_hull_df,
    _rescale_biplot_scores,
    make_network,
    plot_bar,
    plot_heatmap,
    plot_network,
    plot_ordination,
    plot_richness,
    plot_tree,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ps_plot() -> Phyloseq:
    """6 × 4 Phyloseq with sample data and tax table."""
    rng = np.random.default_rng(0)
    taxa = [f"OTU{i + 1}" for i in range(6)]
    samples = [f"S{i + 1}" for i in range(4)]
    df = pd.DataFrame(
        rng.integers(1, 100, size=(6, 4)).astype(float),
        index=taxa,
        columns=samples,
    )
    sam = SampleData(pd.DataFrame({"Group": ["A", "A", "B", "B"]}, index=samples))
    tax = TaxTable(
        pd.DataFrame(
            {
                "Phylum": [f"P{i}" for i in range(6)],
                "Genus": [f"G{i}" for i in range(6)],
            },
            index=taxa,
        )
    )
    return Phyloseq(otu=OtuTable(df), sam=sam, tax=tax)


@pytest.fixture
def ps_with_tree_and_meta() -> Phyloseq:
    """4-taxon Phyloseq with tree, sample data, and tax table."""
    newick = "((OTU1:0.1,OTU2:0.2):0.1,(OTU3:0.15,OTU4:0.05):0.2);"
    tree = PhyTree.from_newick(newick)
    rng = np.random.default_rng(7)
    taxa = ["OTU1", "OTU2", "OTU3", "OTU4"]
    samples = [f"S{i + 1}" for i in range(4)]
    df = pd.DataFrame(
        rng.integers(1, 100, size=(4, 4)).astype(float),
        index=taxa,
        columns=samples,
    )
    sam = SampleData(pd.DataFrame({"Group": ["A", "A", "B", "B"]}, index=samples))
    tax = TaxTable(
        pd.DataFrame({"Phylum": ["Firm", "Firm", "Bact", "Bact"]}, index=taxa)
    )
    return Phyloseq(otu=OtuTable(df), sam=sam, tax=tax, tree=tree)


@pytest.fixture
def pcoa_ord(ps_plot: Phyloseq) -> OrdinationResults:
    return ordinate(ps_plot, method="PCoA", distance="bray")


@pytest.fixture
def ca_ord(ps_plot: Phyloseq) -> OrdinationResults:
    return ordinate(ps_plot, method="CA")


# ---------------------------------------------------------------------------
# plot_bar
# ---------------------------------------------------------------------------


class TestPlotBar:
    def test_returns_ggplot(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot)
        assert isinstance(p, ggplot)

    def test_data_has_required_columns(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot)
        assert "Sample" in p.data.columns
        assert "Abundance" in p.data.columns

    def test_row_count_equals_ntaxa_times_nsamples(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot)
        assert len(p.data) == ps_plot.ntaxa * ps_plot.nsamples

    def test_with_fill(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot, fill="Phylum")
        assert isinstance(p, ggplot)

    def test_with_facet_grid(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot, facet_grid="~ Group")
        assert isinstance(p, ggplot)

    def test_with_title(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot, title="My Bars")
        assert isinstance(p, ggplot)

    def test_custom_x_axis(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot, x="Group")
        assert isinstance(p, ggplot)

    def test_fill_column_absent_does_not_crash(self, ps_plot: Phyloseq) -> None:
        p = plot_bar(ps_plot, fill="NonexistentColumn")
        assert isinstance(p, ggplot)


# ---------------------------------------------------------------------------
# plot_richness
# ---------------------------------------------------------------------------


class TestPlotRichness:
    def test_returns_ggplot(self, ps_plot: Phyloseq) -> None:
        p = plot_richness(ps_plot)
        assert isinstance(p, ggplot)

    def test_data_has_measure_and_value_columns(self, ps_plot: Phyloseq) -> None:
        p = plot_richness(ps_plot)
        assert "Measure" in p.data.columns
        assert "Value" in p.data.columns

    def test_subset_measures_reflected_in_data(self, ps_plot: Phyloseq) -> None:
        p = plot_richness(ps_plot, measures=["Shannon", "Simpson"])
        assert set(p.data["Measure"].unique()) == {"Shannon", "Simpson"}

    def test_with_x_and_color(self, ps_plot: Phyloseq) -> None:
        p = plot_richness(ps_plot, x="Group", color="Group")
        assert isinstance(p, ggplot)

    def test_se_column_present_for_chao1(self, ps_plot: Phyloseq) -> None:
        """estimate_richness produces se.chao1; plot_richness should expose SE."""
        p = plot_richness(ps_plot, measures=["Chao1"])
        assert "SE" in p.data.columns

    def test_with_title(self, ps_plot: Phyloseq) -> None:
        p = plot_richness(ps_plot, title="Alpha Diversity")
        assert isinstance(p, ggplot)

    def test_all_measures_are_valid_names(self, ps_plot: Phyloseq) -> None:
        p = plot_richness(ps_plot)
        measures_in_data = set(p.data["Measure"].unique())
        assert measures_in_data.issubset(set(_ALL_MEASURES))


# ---------------------------------------------------------------------------
# plot_ordination — samples kind
# ---------------------------------------------------------------------------


class TestPlotOrdinationSamples:
    def test_returns_ggplot(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        assert isinstance(plot_ordination(ps_plot, pcoa_ord), ggplot)

    def test_just_df_returns_dataframe(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, pcoa_ord, just_df=True)
        assert isinstance(df, pd.DataFrame)
        assert {"Axis.1", "Axis.2"}.issubset(df.columns)

    def test_sample_count_in_df(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, pcoa_ord, just_df=True)
        assert len(df) == ps_plot.nsamples

    def test_metadata_joined_to_df(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, pcoa_ord, just_df=True)
        assert "Group" in df.columns

    def test_color_mapping(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        assert isinstance(plot_ordination(ps_plot, pcoa_ord, color="Group"), ggplot)

    def test_show_hull(self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults) -> None:
        p = plot_ordination(ps_plot, pcoa_ord, color="Group", show_hull=True)
        assert isinstance(p, ggplot)

    def test_deprecated_type_param(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        with pytest.warns(DeprecationWarning):
            p = plot_ordination(ps_plot, pcoa_ord, type="samples")
        assert isinstance(p, ggplot)


# ---------------------------------------------------------------------------
# plot_ordination — scree kind
# ---------------------------------------------------------------------------


class TestPlotOrdinationScree:
    def test_returns_ggplot(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        assert isinstance(plot_ordination(ps_plot, pcoa_ord, kind="scree"), ggplot)

    def test_just_df_has_axis_and_variance(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, pcoa_ord, kind="scree", just_df=True)
        assert "Axis" in df.columns
        assert "Variance" in df.columns

    def test_scree_variance_nonnegative(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, pcoa_ord, kind="scree", just_df=True)
        assert (df["Variance"] >= 0).all()

    def test_scree_row_count_matches_axes(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, pcoa_ord, kind="scree", just_df=True)
        assert len(df) == len(pcoa_ord.proportion_explained)


# ---------------------------------------------------------------------------
# plot_ordination — taxa kind (requires ordination with feature scores)
# ---------------------------------------------------------------------------


class TestPlotOrdinationTaxa:
    def test_returns_ggplot(self, ps_plot: Phyloseq, ca_ord: OrdinationResults) -> None:
        assert isinstance(plot_ordination(ps_plot, ca_ord, kind="taxa"), ggplot)

    def test_just_df_has_axes(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="taxa", just_df=True)
        assert {"Axis.1", "Axis.2"}.issubset(df.columns)

    def test_taxa_count_in_df(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="taxa", just_df=True)
        assert len(df) == ps_plot.ntaxa

    def test_tax_table_joined_to_df(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="taxa", just_df=True)
        assert "Phylum" in df.columns

    def test_no_features_raises(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        with pytest.raises(pyloseqValidationError):
            plot_ordination(ps_plot, pcoa_ord, kind="taxa")


# ---------------------------------------------------------------------------
# plot_ordination — biplot kind
# ---------------------------------------------------------------------------


class TestPlotOrdinationBiplot:
    def test_returns_ggplot(self, ps_plot: Phyloseq, ca_ord: OrdinationResults) -> None:
        assert isinstance(plot_ordination(ps_plot, ca_ord, kind="biplot"), ggplot)

    def test_just_df_has_type_column(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="biplot", just_df=True)
        assert "_type" in df.columns
        assert set(df["_type"].unique()) == {"samples", "taxa"}

    def test_row_count_is_ntaxa_plus_nsamples(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="biplot", just_df=True)
        assert len(df) == ps_plot.ntaxa + ps_plot.nsamples


# ---------------------------------------------------------------------------
# plot_ordination — split kind
# ---------------------------------------------------------------------------


class TestPlotOrdinationSplit:
    def test_returns_ggplot(self, ps_plot: Phyloseq, ca_ord: OrdinationResults) -> None:
        assert isinstance(plot_ordination(ps_plot, ca_ord, kind="split"), ggplot)

    def test_just_df_has_panel_column(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="split", just_df=True)
        assert "_panel" in df.columns

    def test_both_panels_present(
        self, ps_plot: Phyloseq, ca_ord: OrdinationResults
    ) -> None:
        df = plot_ordination(ps_plot, ca_ord, kind="split", just_df=True)
        panels = set(df["_panel"].unique())
        assert {"Samples", "Taxa"}.issubset(panels)


# ---------------------------------------------------------------------------
# plot_ordination — error cases
# ---------------------------------------------------------------------------


class TestPlotOrdinationErrors:
    def test_unknown_kind_raises(
        self, ps_plot: Phyloseq, pcoa_ord: OrdinationResults
    ) -> None:
        with pytest.raises(pyloseqValidationError):
            plot_ordination(ps_plot, pcoa_ord, kind="unknown_kind")


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


class TestPlotHeatmap:
    def test_returns_ggplot(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method="PCoA")
        assert isinstance(p, ggplot)

    def test_data_has_sample_otu_abundance(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method="PCoA")
        assert {"Sample", "OTU", "Abundance"}.issubset(p.data.columns)

    def test_no_transform(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method="PCoA", trans=None)
        assert isinstance(p, ggplot)

    def test_log4_zeros_become_nan(self, ps_plot: Phyloseq) -> None:
        """log4 transform should produce NaN for zero cells (they use na_value colour)."""
        p = plot_heatmap(ps_plot, method="PCoA", trans="log4")
        original_zeros = (ps_plot.otu_table.to_dataframe() == 0).values.sum()
        if original_zeros > 0:
            assert p.data["Abundance"].isna().sum() >= original_zeros

    def test_bad_trans_raises(self, ps_plot: Phyloseq) -> None:
        with pytest.raises(pyloseqValidationError):
            plot_heatmap(ps_plot, method="PCoA", trans="sqrt")

    def test_with_title(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method="PCoA", title="Heatmap")
        assert isinstance(p, ggplot)

    def test_row_count(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method="PCoA")
        assert len(p.data) == ps_plot.ntaxa * ps_plot.nsamples

    def test_method_none_returns_ggplot(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None)
        assert isinstance(p, ggplot)

    def test_method_none_preserves_sample_order(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None)
        cats = list(p.data["Sample"].cat.categories)
        assert cats == list(ps_plot.sample_names)

    def test_method_none_preserves_taxa_order(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None)
        cats = list(p.data["OTU"].cat.categories)
        assert cats == list(ps_plot.taxa_names)

    def test_label_adds_scale(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None, label="Group")
        assert isinstance(p, ggplot)
        assert any(isinstance(s, ScaleXDiscrete) for s in p.scales)

    def test_label_maps_sample_names_to_variable(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None, label="Group")
        scale = next(s for s in p.scales if isinstance(s, ScaleXDiscrete))
        # labels dict maps each sample name to its Group value
        assert isinstance(scale.labels, dict)
        assert set(scale.labels.values()) == {"A", "B"}

    def test_label_missing_warns(self, ps_plot: Phyloseq) -> None:
        with pytest.warns(UserWarning, match="not found in sample_data"):
            plot_heatmap(ps_plot, method=None, label="NoSuchColumn")

    def test_taxa_label_adds_scale(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None, taxa_label="Phylum")
        assert isinstance(p, ggplot)
        assert any(isinstance(s, ScaleYDiscrete) for s in p.scales)

    def test_taxa_label_maps_taxa_names_to_rank(self, ps_plot: Phyloseq) -> None:
        p = plot_heatmap(ps_plot, method=None, taxa_label="Phylum")
        scale = next(s for s in p.scales if isinstance(s, ScaleYDiscrete))
        # labels dict maps each OTU name to its Phylum value
        assert isinstance(scale.labels, dict)
        assert set(scale.labels.values()) == {f"P{i}" for i in range(6)}

    def test_taxa_label_missing_warns(self, ps_plot: Phyloseq) -> None:
        with pytest.warns(UserWarning, match="not found in tax_table"):
            plot_heatmap(ps_plot, method=None, taxa_label="NoSuchRank")


# ---------------------------------------------------------------------------
# plot_tree
# ---------------------------------------------------------------------------


class TestPlotTree:
    def test_treeonly_returns_ggplot(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(plot_tree(ps_with_tree_and_meta, method="treeonly"), ggplot)

    def test_sampledodge_returns_ggplot(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(
            plot_tree(ps_with_tree_and_meta, method="sampledodge"), ggplot
        )

    def test_with_color(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(plot_tree(ps_with_tree_and_meta, color="Group"), ggplot)

    def test_with_label_tips(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(plot_tree(ps_with_tree_and_meta, label_tips="Phylum"), ggplot)

    def test_ladderize_right(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(plot_tree(ps_with_tree_and_meta, ladderize=True), ggplot)

    def test_ladderize_left(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(plot_tree(ps_with_tree_and_meta, ladderize="left"), ggplot)

    def test_left_justify(self, ps_with_tree_and_meta: Phyloseq) -> None:
        assert isinstance(plot_tree(ps_with_tree_and_meta, justify="left"), ggplot)

    def test_no_tree_raises(self, ps_plot: Phyloseq) -> None:
        with pytest.raises(pyloseqValidationError):
            plot_tree(ps_plot)

    def test_bad_method_raises(self, ps_with_tree_and_meta: Phyloseq) -> None:
        with pytest.raises(pyloseqValidationError):
            plot_tree(ps_with_tree_and_meta, method="badmethod")

    def test_bad_justify_raises(self, ps_with_tree_and_meta: Phyloseq) -> None:
        with pytest.raises(pyloseqValidationError):
            plot_tree(ps_with_tree_and_meta, justify="center")

    def test_label_tips_no_tax_table_raises(self) -> None:
        rng = np.random.default_rng(0)
        taxa = ["A", "B", "C"]
        tree = PhyTree.from_newick("(A:0.1,B:0.2,C:0.3);")
        df = pd.DataFrame(
            rng.integers(1, 10, size=(3, 2)).astype(float),
            index=taxa,
            columns=["S1", "S2"],
        )
        ps = Phyloseq(otu=OtuTable(df), tree=tree)
        with pytest.raises(pyloseqValidationError):
            plot_tree(ps, label_tips="Genus")

    def test_with_title(self, ps_with_tree_and_meta: Phyloseq) -> None:
        p = plot_tree(ps_with_tree_and_meta, title="My Tree")
        assert isinstance(p, ggplot)


# ---------------------------------------------------------------------------
# make_network
# ---------------------------------------------------------------------------

networkx = pytest.importorskip("networkx", reason="networkx not installed")


class TestMakeNetwork:
    def test_returns_graph(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=0.99)
        assert isinstance(g, nx.Graph)

    def test_nodes_are_sample_names(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=0.99)
        assert set(g.nodes) <= set(ps_plot.sample_names)

    def test_keep_isolates_true_preserves_all_nodes(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=0.0, keep_isolates=True)
        assert g.number_of_nodes() == ps_plot.nsamples

    def test_zero_max_dist_no_edges(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=0.0, keep_isolates=True)
        assert g.number_of_edges() == 0

    def test_keep_isolates_false_removes_disconnected(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=0.0, keep_isolates=False)
        assert g.number_of_nodes() == 0

    def test_high_max_dist_connects_all(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=1.0, keep_isolates=True)
        assert g.number_of_nodes() == ps_plot.nsamples

    def test_sample_metadata_attached_to_nodes(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=1.0, keep_isolates=True)
        for node in g.nodes:
            assert "Group" in g.nodes[node]

    def test_edge_weights_in_unit_range(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, max_dist=1.0, keep_isolates=True)
        for _, _, data in g.edges(data=True):
            assert 0.0 <= data["weight"] <= 1.0

    def test_taxa_kind(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, kind="taxa", max_dist=1.0)
        assert isinstance(g, nx.Graph)
        assert set(g.nodes) <= set(ps_plot.taxa_names)

    def test_tax_table_metadata_attached_for_taxa_kind(self, ps_plot: Phyloseq) -> None:
        g = make_network(ps_plot, kind="taxa", max_dist=1.0, keep_isolates=True)
        for node in g.nodes:
            assert "Phylum" in g.nodes[node]

    def test_deprecated_type_param(self, ps_plot: Phyloseq) -> None:
        with pytest.warns(DeprecationWarning):
            g = make_network(ps_plot, type="samples", max_dist=1.0)
        assert isinstance(g, nx.Graph)


# ---------------------------------------------------------------------------
# plot_network
# ---------------------------------------------------------------------------


class TestPlotNetwork:
    @pytest.fixture
    def simple_graph(self, ps_plot: Phyloseq) -> object:
        return make_network(ps_plot, max_dist=1.0, keep_isolates=True)

    def test_returns_ggplot(self, simple_graph: Any, ps_plot: Phyloseq) -> None:
        assert isinstance(plot_network(simple_graph, ps_plot), ggplot)

    def test_with_color(self, simple_graph: Any, ps_plot: Phyloseq) -> None:
        assert isinstance(plot_network(simple_graph, ps_plot, color="Group"), ggplot)

    def test_with_label(self, simple_graph: Any, ps_plot: Phyloseq) -> None:
        assert isinstance(plot_network(simple_graph, ps_plot, label="node"), ggplot)

    def test_with_title(self, simple_graph: Any, ps_plot: Phyloseq) -> None:
        assert isinstance(plot_network(simple_graph, ps_plot, title="Network"), ggplot)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestRescaleBiplotScores:
    def test_empty_feature_array_unchanged(self) -> None:
        sample_xy = np.array([[1.0, 0.0], [-1.0, 0.0]])
        feature_xy = np.zeros((0, 2))
        result = _rescale_biplot_scores(sample_xy, feature_xy)
        assert result.shape == (0, 2)

    def test_rescales_to_sample_span(self) -> None:
        sample_xy = np.array([[10.0, 0.0], [-10.0, 0.0]])
        feature_xy = np.array([[1.0, 0.0], [-1.0, 0.0]])
        result = _rescale_biplot_scores(sample_xy, feature_xy)
        np.testing.assert_allclose(np.abs(result).max(), 10.0)

    def test_identity_when_spans_equal(self) -> None:
        xy = np.array([[5.0, 3.0], [-5.0, -3.0]])
        result = _rescale_biplot_scores(xy, xy.copy())
        np.testing.assert_allclose(result, xy)

    def test_empty_sample_array_returns_features_unchanged(self) -> None:
        sample_xy = np.zeros((0, 2))
        feature_xy = np.array([[1.0, 2.0]])
        result = _rescale_biplot_scores(sample_xy, feature_xy)
        np.testing.assert_allclose(result, feature_xy)


class TestConvexHullDf:
    def test_too_few_points_returns_empty(self) -> None:
        df = pd.DataFrame(
            {"Axis.1": [0.0, 1.0], "Axis.2": [0.0, 1.0], "Group": ["A", "A"]}
        )
        assert _convex_hull_df(df, "Group").empty

    def test_three_points_returns_hull(self) -> None:
        df = pd.DataFrame(
            {
                "Axis.1": [0.0, 1.0, 0.5],
                "Axis.2": [0.0, 0.0, 1.0],
                "Group": ["A", "A", "A"],
            }
        )
        result = _convex_hull_df(df, "Group")
        assert not result.empty
        assert {"Axis.1", "Axis.2", "Group"}.issubset(result.columns)

    def test_multiple_groups(self) -> None:
        df = pd.DataFrame(
            {
                "Axis.1": [0.0, 1.0, 0.5, 5.0, 6.0, 5.5],
                "Axis.2": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                "Group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        result = _convex_hull_df(df, "Group")
        assert {"A", "B"}.issubset(set(result["Group"].unique()))

    def test_collinear_points_skipped(self) -> None:
        df = pd.DataFrame(
            {
                "Axis.1": [0.0, 1.0, 2.0, 3.0],
                "Axis.2": [0.0, 1.0, 2.0, 3.0],
                "Group": ["A"] * 4,
            }
        )
        result = _convex_hull_df(df, "Group")
        assert result.empty
