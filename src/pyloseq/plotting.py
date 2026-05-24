"""Plotting functions mirroring R phyloseq's visualization API.

All functions return ``plotnine.ggplot`` objects (except :func:`make_network`
which returns a ``networkx.Graph``).  The underlying data is always available
via the plot's ``.data`` attribute.

R reference: phyloseq plot_bar, plot_richness, plot_ordination, plot_heatmap,
             plot_tree, make_network, plot_network
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyloseq._exceptions import pyloseqValidationError

if TYPE_CHECKING:
    from pyloseq._phyloseq import Phyloseq


# ---------------------------------------------------------------------------
# plot_bar
# ---------------------------------------------------------------------------


def plot_bar(
    ps: Phyloseq,
    x: str = "Sample",
    y: str = "Abundance",
    fill: str | None = None,
    facet_grid: str | None = None,
    title: str | None = None,
) -> Any:
    """Stacked bar chart of OTU/feature abundances.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    x:
        Column in the melted data to use as x-axis (default ``"Sample"``).
    y:
        Column for bar height (default ``"Abundance"``).
    fill:
        Column for bar fill colour (e.g. ``"Phylum"``).
    facet_grid:
        Facet formula string (e.g. ``"~ SampleType"``).
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_bar(physeq, x, y, fill, facet_grid, title)
    """
    from plotnine import (
        aes,
        element_text,
        geom_bar,
        ggplot,
        labs,
        theme,
    )
    from plotnine import facet_grid as pg_facet_grid

    from pyloseq._manipulation import psmelt  # noqa: PLC0415

    long_df = psmelt(ps)

    mapping: dict[str, str] = {"x": x, "y": y}
    if fill:
        mapping["fill"] = fill

    p = (
        ggplot(long_df, aes(**mapping))
        + geom_bar(stat="identity", position="stack")
        + theme(axis_text_x=element_text(rotation=90, hjust=1))
    )

    if facet_grid:
        p = p + pg_facet_grid(facet_grid)

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# plot_richness
# ---------------------------------------------------------------------------


def plot_richness(
    ps: Phyloseq,
    x: str = "samples",
    color: str | None = None,
    measures: list[str] | None = None,
    title: str | None = None,
) -> Any:
    """Alpha diversity box-and-point plots, faceted by measure.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    x:
        Sample metadata column for x-axis.  Use ``"samples"`` (default) to
        put individual samples on x.
    color:
        Sample metadata column for point colour.
    measures:
        Diversity measures to include; default is all.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_richness(physeq, x, color, measures, title)
    """
    from plotnine import (
        aes,
        element_text,
        facet_wrap,
        geom_boxplot,
        geom_point,
        ggplot,
        labs,
        theme,
    )

    from pyloseq._diversity import estimate_richness  # noqa: PLC0415

    rich_df = estimate_richness(ps, measures=measures)

    # Join sample metadata
    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        rich_df = rich_df.join(sam_df, how="left")

    if x == "samples":
        rich_df["_sample"] = rich_df.index
        x_col = "_sample"
    else:
        x_col = x

    # Melt to long form for faceting
    measure_cols = [
        c
        for c in rich_df.columns
        if c
        in (
            measures
            or [
                "Observed",
                "Chao1",
                "se.chao1",
                "ACE",
                "se.ACE",
                "Shannon",
                "Simpson",
                "InvSimpson",
                "Fisher",
            ]
        )
    ]
    id_vars = [c for c in rich_df.columns if c not in measure_cols]
    long = rich_df.reset_index().melt(
        id_vars=["index"] + [c for c in id_vars if c != "index"],
        value_vars=measure_cols,
        var_name="Measure",
        value_name="Value",
    )

    mapping: dict[str, str] = {"x": x_col if x_col in long.columns else "index", "y": "Value"}
    if color and color in long.columns:
        mapping["color"] = color

    p = (
        ggplot(long, aes(**mapping))
        + geom_boxplot(alpha=0.5)
        + geom_point(size=2)
        + facet_wrap("~Measure", scales="free_y")
        + theme(axis_text_x=element_text(rotation=90, hjust=1))
    )

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# plot_ordination
# ---------------------------------------------------------------------------


def plot_ordination(
    ps: Phyloseq,
    ord: Any,
    type: str = "samples",
    color: str | None = None,
    shape: str | None = None,
    label: str | None = None,
    title: str | None = None,
) -> Any:
    """Scatter plot of ordination results.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    ord:
        ``skbio.stats.ordination.OrdinationResults`` from :func:`ordinate`.
    type:
        One of ``"samples"``, ``"taxa"``, ``"biplot"``, ``"split"``,
        ``"scree"``.
    color:
        Sample/taxa metadata column for point colour.
    shape:
        Sample metadata column for point shape.
    label:
        Column to annotate points with text labels.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_ordination(physeq, ordination, type, color, shape, label, title)
    """
    from plotnine import (
        aes,
        geom_point,
        geom_text,
        ggplot,
        labs,
        xlab,
        ylab,
    )

    if type == "scree":
        return _plot_scree(ord, title=title)

    if type in ("samples", "biplot"):
        plot_df = pd.DataFrame(
            ord.samples.values[:, :2], index=ord.samples.index, columns=["Axis.1", "Axis.2"]
        )
        plot_df.index.name = "sample_id"

        if ps.sample_data is not None:
            sam_df = ps.sample_data.to_frame()
            plot_df = plot_df.join(sam_df, how="left")

        if type == "biplot" and ord.features is not None:
            feat_df = pd.DataFrame(
                ord.features.values[:, :2],
                index=ord.features.index,
                columns=["Axis.1", "Axis.2"],
            )
            if ps.tax_table is not None:
                feat_df = feat_df.join(ps.tax_table.to_frame(), how="left")
            feat_df["_type"] = "taxa"
            plot_df["_type"] = "samples"
            plot_df = pd.concat([plot_df, feat_df], axis=0)

    elif type == "taxa":
        if ord.features is None:
            raise pyloseqValidationError(
                "Ordination result has no 'features' (taxa) scores. "
                "Use type='samples' or re-run ordinate with a method that produces taxa scores."
            )
        plot_df = pd.DataFrame(
            ord.features.values[:, :2],
            index=ord.features.index,
            columns=["Axis.1", "Axis.2"],
        )
        if ps.tax_table is not None:
            plot_df = plot_df.join(ps.tax_table.to_frame(), how="left")

    elif type == "split":
        # Two panels: samples and taxa side by side
        return _plot_split(ps, ord, color=color, shape=shape)

    else:
        raise pyloseqValidationError(
            f"Unknown plot_ordination type: '{type}'. "
            "Use: 'samples', 'taxa', 'biplot', 'split', 'scree'."
        )

    plot_df = plot_df.reset_index()

    mapping: dict[str, str] = {"x": "Axis.1", "y": "Axis.2"}
    if color and color in plot_df.columns:
        mapping["color"] = color
    if shape and shape in plot_df.columns:
        mapping["shape"] = shape

    # Axis labels with % variance if available
    def _axis_label(i: int) -> str:
        name = f"Axis.{i}"
        if (
            ord.proportion_explained is not None
            and not ord.proportion_explained.isna().all()
            and len(ord.proportion_explained) >= i
        ):
            pct = 100 * ord.proportion_explained.iloc[i - 1]
            return f"{name} [{pct:.1f}%]"
        return name

    p = (
        ggplot(plot_df, aes(**mapping))
        + geom_point(size=3)
        + xlab(_axis_label(1))
        + ylab(_axis_label(2))
    )

    if label and label in plot_df.columns:
        p = p + geom_text(aes(label=label), nudge_y=0.01, size=7)

    if title:
        p = p + labs(title=title)

    return p


def _plot_scree(ord: Any, title: str | None = None) -> Any:
    """Scree plot of eigenvalues / proportion explained."""
    from plotnine import (
        aes,
        geom_line,
        geom_point,
        ggplot,
        labs,
        xlab,
        ylab,
    )

    if ord.proportion_explained is None or ord.proportion_explained.isna().all():
        raise pyloseqValidationError(
            "Ordination result has no proportion_explained; scree plot unavailable."
        )
    df = pd.DataFrame(
        {
            "Axis": range(1, len(ord.proportion_explained) + 1),
            "Variance": ord.proportion_explained.values * 100,
        }
    )
    p = (
        ggplot(df, aes("Axis", "Variance"))
        + geom_line()
        + geom_point(size=3)
        + xlab("Axis")
        + ylab("% Variance Explained")
    )
    if title:
        p = p + labs(title=title)
    return p


def _plot_split(
    ps: Phyloseq,
    ord: Any,
    color: str | None = None,
    shape: str | None = None,
) -> Any:
    """Split biplot: samples and taxa in side-by-side facets."""
    from plotnine import aes, facet_wrap, geom_point, ggplot

    frames = []
    sam_df = pd.DataFrame(
        ord.samples.values[:, :2], index=ord.samples.index, columns=["Axis.1", "Axis.2"]
    )
    sam_df["_panel"] = "Samples"
    if ps.sample_data is not None:
        sam_df = sam_df.join(ps.sample_data.to_frame(), how="left")
    frames.append(sam_df)

    if ord.features is not None:
        feat_df = pd.DataFrame(
            ord.features.values[:, :2], index=ord.features.index, columns=["Axis.1", "Axis.2"]
        )
        feat_df["_panel"] = "Taxa"
        if ps.tax_table is not None:
            feat_df = feat_df.join(ps.tax_table.to_frame(), how="left")
        frames.append(feat_df)

    combined = pd.concat(frames).reset_index()
    mapping: dict[str, str] = {"x": "Axis.1", "y": "Axis.2"}
    if color and color in combined.columns:
        mapping["color"] = color
    if shape and shape in combined.columns:
        mapping["shape"] = shape
    return ggplot(combined, aes(**mapping)) + geom_point(size=3) + facet_wrap("~_panel")


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    ps: Phyloseq,
    method: str = "NMDS",
    distance: str = "bray",
    trans: str | None = None,
    low: str = "#000033",
    high: str = "#66CCFF",
    title: str | None = None,
) -> Any:
    """Abundance heatmap with samples and taxa reordered by ordination.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    method:
        Ordination method used to reorder samples.
    distance:
        Distance metric for ordination.
    trans:
        Transformation applied to abundances before plotting.
        ``"log4"`` computes ``log4(x + 1)``; ``None`` uses raw counts.
    low, high:
        Gradient colour endpoints.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_heatmap(physeq, method, distance, trans, low, high, title)
    """
    from plotnine import (
        aes,
        element_text,
        geom_tile,
        ggplot,
        labs,
        scale_fill_gradient,
        theme,
    )

    from pyloseq._manipulation import psmelt  # noqa: PLC0415
    from pyloseq._ordination import ordinate  # noqa: PLC0415

    # Ordinate to get sample ordering
    try:
        ord_result = ordinate(ps, method=method, distance=distance)
        sample_order = list(ord_result.samples.sort_values("Axis.1").index)
    except Exception:
        sample_order = list(ps.sample_names)

    long_df = psmelt(ps)

    # Apply transformation
    if trans == "log4":
        long_df["Abundance"] = np.log(long_df["Abundance"] + 1) / np.log(4)
    elif trans is not None:
        raise pyloseqValidationError(f"Unknown trans '{trans}'. Use 'log4' or None.")

    # Order samples by ordination axis
    long_df["Sample"] = pd.Categorical(long_df["Sample"], categories=sample_order, ordered=True)

    p = (
        ggplot(long_df, aes(x="Sample", y="OTU", fill="Abundance"))
        + geom_tile()
        + scale_fill_gradient(low=low, high=high)
        + theme(axis_text_x=element_text(rotation=90, hjust=1))
    )

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# make_network / plot_network
# ---------------------------------------------------------------------------


def make_network(
    ps: Phyloseq,
    type: str = "samples",
    distance: str = "jaccard",
    max_dist: float = 0.4,
    keep_isolates: bool = False,
) -> Any:
    """Build a sample (or taxa) network based on a distance threshold.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    type:
        ``"samples"`` (default) or ``"taxa"``.
    distance:
        Distance metric (see :func:`pyloseq.distance`).
    max_dist:
        Maximum distance for an edge to be drawn.
    keep_isolates:
        If ``False`` (default), remove nodes with no edges.

    Returns
    -------
    networkx.Graph

    R reference: make_network(physeq, type, distance, max.dist, keep.isolates)
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "make_network requires networkx. Install it with: pip install networkx"
        ) from e

    from pyloseq._distances import distance as _distance  # noqa: PLC0415

    dm = _distance(ps, distance, type=type)
    ids = list(dm.ids)
    data = np.array(dm.data)

    g = nx.Graph()
    g.add_nodes_from(ids)

    for i, u in enumerate(ids):
        for j, v in enumerate(ids):
            if j <= i:
                continue
            if data[i, j] <= max_dist:
                g.add_edge(u, v, weight=float(data[i, j]))

    if not keep_isolates:
        isolates = list(nx.isolates(g))
        g.remove_nodes_from(isolates)

    # Attach sample metadata as node attributes
    if type == "samples" and ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        for node in g.nodes:
            if node in sam_df.index:
                for col in sam_df.columns:
                    g.nodes[node][col] = sam_df.loc[node, col]

    return g


def plot_network(
    g: Any,
    ps: Phyloseq,
    color: str | None = None,
    layout: str = "fruchterman_reingold",
    title: str | None = None,
) -> Any:
    """Plot a network graph as a ggplot scatter.

    Parameters
    ----------
    g:
        ``networkx.Graph`` from :func:`make_network`.
    ps:
        ``Phyloseq`` object (used for metadata annotations).
    color:
        Node attribute column for colour.
    layout:
        NetworkX layout algorithm name (e.g. ``"fruchterman_reingold"``).
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_network(ig, physeq, color, layout, title)
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "plot_network requires networkx. Install it with: pip install networkx"
        ) from e

    from plotnine import aes, geom_point, geom_segment, ggplot, labs

    layout_fn = getattr(nx, f"{layout}_layout", nx.spring_layout)
    pos = layout_fn(g)

    # Nodes data frame
    rows = []
    for node, (x, y) in pos.items():
        row: dict[str, Any] = {"node": node, "x": x, "y": y}
        row.update(g.nodes[node])
        rows.append(row)
    nodes_df = pd.DataFrame(rows)

    # Edges data frame
    edge_rows = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_rows.append({"x": x0, "y": y0, "xend": x1, "yend": y1})
    edges_df = (
        pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(columns=["x", "y", "xend", "yend"])
    )

    mapping: dict[str, str] = {"x": "x", "y": "y"}
    if color and color in nodes_df.columns:
        mapping["color"] = color

    p = ggplot(nodes_df, aes(**mapping))

    if not edges_df.empty:
        p = p + geom_segment(
            data=edges_df,
            mapping=aes(x="x", y="y", xend="xend", yend="yend"),
            color="grey70",
        )

    p = p + geom_point(size=5)

    if title:
        p = p + labs(title=title)

    return p
