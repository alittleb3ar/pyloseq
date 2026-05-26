"""Tests for plotting functions: plot_bar, plot_richness, plot_ordination, plot_heatmap."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyloseq
from pyloseq import OtuTable, Phyloseq, SampleData, TaxTable, ordinate


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


# ===========================================================================
# Plotting
# ===========================================================================


class TestPlotting:
    def test_plot_bar_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = pyloseq.plot_bar(ps)
        assert isinstance(p, ggplot)

    def test_plot_bar_with_fill(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        tax_df = pd.DataFrame(
            {"Phylum": ["A", "A", "B", "B", "C", "C"]},
            index=[f"OTU{i + 1}" for i in range(6)],
        )
        ps2 = Phyloseq(otu=ps.otu_table.copy(), tax=TaxTable(tax_df))
        p = pyloseq.plot_bar(ps2, fill="Phylum")
        assert isinstance(p, ggplot)

    def test_plot_richness_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = pyloseq.plot_richness(ps, measures=["Observed", "Shannon"])
        assert isinstance(p, ggplot)

    def test_plot_ordination_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        ord_result = ordinate(ps, method="PCoA", distance="bray")
        p = pyloseq.plot_ordination(ps, ord_result, type="samples")
        assert isinstance(p, ggplot)

    def test_plot_ordination_scree(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        ord_result = ordinate(ps, method="PCoA", distance="euclidean")
        p = pyloseq.plot_ordination(ps, ord_result, type="scree")
        assert isinstance(p, ggplot)

    def test_plot_heatmap_returns_ggplot(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = pyloseq.plot_heatmap(ps, method="PCoA", distance="bray")
        assert isinstance(p, ggplot)

    def test_plot_heatmap_log4_trans(self) -> None:
        from plotnine import ggplot

        ps = _make_ps()
        p = pyloseq.plot_heatmap(ps, trans="log4")
        assert isinstance(p, ggplot)


# ===========================================================================
# plot_heatmap warns on bad ordination method
# ===========================================================================


class TestPlotHeatmapOrdWarning:
    def test_plot_heatmap_bad_method_warns(self) -> None:
        ps = _make_ps()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = pyloseq.plot_heatmap(ps, method="GARBAGE", distance="bray")
            assert any("ordination failed" in str(x.message).lower() for x in w)
        assert p is not None
