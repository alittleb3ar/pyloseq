"""Plotting functions mirroring R phyloseq's visualization API.

All functions return ``plotnine.ggplot`` objects (except :func:`make_network`
which returns a ``networkx.Graph``).  The underlying data is always available
via the plot's ``.data`` attribute.

R reference: phyloseq plot_bar, plot_richness, plot_ordination, plot_heatmap,
             plot_tree, make_network, plot_network
"""

from __future__ import annotations

import warnings
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

    Bars are ordered by ``x`` and stacked deterministically by ``fill`` so
    that fill segments line up consistently across samples (matching R
    phyloseq's behaviour).

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
    from plotnine import aes, element_text
    from plotnine import facet_grid as pg_facet_grid
    from plotnine import geom_bar, ggplot, labs, theme

    from pyloseq._manipulation import psmelt  # noqa: PLC0415

    long_df = psmelt(ps)

    # Deterministic stacking order: sort by x then fill so that stacked
    # segments are consistent across samples.  R orders the melted frame
    # by the fill taxon before drawing.
    sort_keys = [x]
    if fill and fill in long_df.columns:
        sort_keys.append(fill)
    sort_keys = [k for k in sort_keys if k in long_df.columns]
    if sort_keys:
        long_df = long_df.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    mapping: dict[str, str] = {"x": x, "y": y}
    if fill:
        mapping["fill"] = fill

    p = (
        ggplot(long_df, aes(**mapping))
        # Outlined segments (one per OTU) like phyloseq; group by OTU so each
        # feature is its own rectangle within the stack.
        + geom_bar(
            stat="identity",
            position="stack",
            mapping=aes(group="OTU") if "OTU" in long_df.columns else None,
            color="black",
            size=0.1,
        )
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

    Standard-error whiskers are drawn for any measure that has a matching
    ``se.<measure>`` column from :func:`estimate_richness` (e.g. ``se.chao1``,
    ``se.ACE``), matching R phyloseq.

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
    from plotnine import (aes, element_text, facet_wrap, geom_boxplot,
                          geom_errorbar, geom_point, ggplot, labs, theme)

    from pyloseq._diversity import (_ALL_MEASURES,  # noqa: PLC0415
                                    estimate_richness)

    rich_df = estimate_richness(ps, measures=measures)

    # Separate the value columns from their standard-error partners.
    se_cols = [c for c in rich_df.columns if c.lower().startswith("se.")]
    se_map = {c[3:]: c for c in se_cols}  # measure name -> se column

    requested = measures or _ALL_MEASURES
    measure_cols = [
        c
        for c in rich_df.columns
        if c not in se_cols and c in requested
    ]

    # Join sample metadata
    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        rich_df = rich_df.join(sam_df, how="left")

    # Use a collision-proof name for the sample id rather than the literal
    # string "index", which can clash with a metadata column.
    sample_id_col = "_sample_id"
    rich_df[sample_id_col] = rich_df.index

    if x == "samples":
        x_col = sample_id_col
    else:
        x_col = x

    id_vars = [c for c in rich_df.columns if c not in measure_cols and c not in se_cols]

    # Melt the measure values to long form.
    long = rich_df.melt(
        id_vars=id_vars,
        value_vars=measure_cols,
        var_name="Measure",
        value_name="Value",
    )

    # Melt the SE values in parallel (if any) and merge them back, aligned by
    # sample id + measure.
    if se_map:
        se_long_frames = []
        for measure, se_col in se_map.items():
            if measure not in measure_cols:
                continue
            sub = rich_df[[sample_id_col, se_col]].copy()
            sub["Measure"] = measure
            sub = sub.rename(columns={se_col: "SE"})
            se_long_frames.append(sub)
        if se_long_frames:
            se_long = pd.concat(se_long_frames, ignore_index=True)
            long = long.merge(se_long, on=[sample_id_col, "Measure"], how="left")
    if "SE" not in long.columns:
        long["SE"] = np.nan

    long["ymin"] = long["Value"] - long["SE"]
    long["ymax"] = long["Value"] + long["SE"]

    mapping: dict[str, str] = {
        "x": x_col if x_col in long.columns else sample_id_col,
        "y": "Value",
    }
    if color and color in long.columns:
        mapping["color"] = color

    p = ggplot(long, aes(**mapping))

    # Boxplot only makes sense when x groups multiple samples; still harmless
    # for per-sample x (renders as a degenerate box) and matches R, which
    # always layers it.
    p = p + geom_boxplot(alpha=0.5)

    # Error bars where SE is available.
    if long["SE"].notna().any():
        p = p + geom_errorbar(
            aes(ymin="ymin", ymax="ymax"),
            width=0.2,
        )

    p = (
        p
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


def _rescale_biplot_scores(
    sample_xy: np.ndarray,
    feature_xy: np.ndarray,
) -> np.ndarray:
    """Rescale feature (taxa) scores so they are comparable to sample scores.

    Uses the classic biplot heuristic: scale the feature coordinates so their
    spread matches the sample spread, so taxa arrows/points sit on the same
    visual scale as the samples instead of collapsing near the origin or
    flying off-axis.
    """
    if feature_xy.size == 0 or sample_xy.size == 0:
        return feature_xy
    samp_span = np.nanmax(np.abs(sample_xy)) or 1.0
    feat_span = np.nanmax(np.abs(feature_xy)) or 1.0
    return feature_xy * (samp_span / feat_span)


def plot_ordination(
    ps: Phyloseq,
    ord: Any,
    kind: str = "samples",
    color: str | None = None,
    shape: str | None = None,
    label: str | None = None,
    title: str | None = None,
    show_hull: bool = False,
    just_df: bool = False,
    **kwargs: Any,
) -> Any:
    """Scatter plot of ordination results.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    ord:
        ``skbio.stats.ordination.OrdinationResults`` from :func:`ordinate`.
    kind:
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
    show_hull:
        If ``True``, shade a convex hull behind each colour group (samples
        and split kinds only).  Off by default; this is not phyloseq
        behaviour but is offered as a convenience.
    just_df:
        If ``True``, return the assembled plotting ``DataFrame`` instead of a
        ggplot (mirrors R phyloseq's ``justDF=TRUE``).

    Returns
    -------
    plotnine.ggplot or pandas.DataFrame

    R reference: plot_ordination(physeq, ordination, type, color, shape,
                                 label, title, justDF)
    """
    if "type" in kwargs:
        warnings.warn(
            "The 'type' parameter is deprecated; use 'kind' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kind = kwargs.pop("type")

    from plotnine import aes, geom_point, geom_text, ggplot, labs, xlab, ylab

    if kind == "scree":
        if just_df:
            return pd.DataFrame(
                {
                    "Axis": range(1, len(ord.proportion_explained) + 1),
                    "Variance": ord.proportion_explained.values * 100,
                }
            )
        return _plot_scree(ord, title=title)

    if kind in ("samples", "biplot"):
        plot_df = pd.DataFrame(
            ord.samples.values[:, :2],
            index=ord.samples.index,
            columns=["Axis.1", "Axis.2"],
        )
        plot_df.index.name = "sample_id"

        if ps.sample_data is not None:
            sam_df = ps.sample_data.to_frame()
            plot_df = plot_df.join(sam_df, how="left")

        if kind == "biplot" and ord.features is not None:
            feat_xy = _rescale_biplot_scores(
                ord.samples.values[:, :2],
                ord.features.values[:, :2],
            )
            feat_df = pd.DataFrame(
                feat_xy,
                index=ord.features.index,
                columns=["Axis.1", "Axis.2"],
            )
            if ps.tax_table is not None:
                feat_df = feat_df.join(ps.tax_table.to_frame(), how="left")
            feat_df["_type"] = "taxa"
            plot_df["_type"] = "samples"
            plot_df = pd.concat([plot_df, feat_df], axis=0)

    elif kind == "taxa":
        if ord.features is None:
            raise pyloseqValidationError(
                "Ordination result has no 'features' (taxa) scores. "
                "Use kind='samples' or re-run ordinate with a method that "
                "produces taxa scores."
            )
        plot_df = pd.DataFrame(
            ord.features.values[:, :2],
            index=ord.features.index,
            columns=["Axis.1", "Axis.2"],
        )
        if ps.tax_table is not None:
            plot_df = plot_df.join(ps.tax_table.to_frame(), how="left")

    elif kind == "split":
        if just_df:
            return _split_df(ps, ord)
        return _plot_split(
            ps, ord, color=color, shape=shape, label=label,
            title=title, show_hull=show_hull,
        )

    else:
        raise pyloseqValidationError(
            f"Unknown plot_ordination kind: '{kind}'. "
            "Use: 'samples', 'taxa', 'biplot', 'split', 'scree'."
        )

    plot_df = plot_df.reset_index()

    if just_df:
        return plot_df

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

    from plotnine import geom_polygon  # noqa: PLC0415

    p = ggplot(plot_df, aes(**mapping)) + xlab(_axis_label(1)) + ylab(_axis_label(2))

    # Convex hull shading per group (opt-in; samples kind only since biplot
    # mixes taxa rows into the frame).
    if show_hull and color and color in plot_df.columns and kind == "samples":
        hull_data = _convex_hull_df(plot_df.dropna(subset=[color]), color)
        if not hull_data.empty:
            p = p + geom_polygon(
                data=hull_data,
                mapping=aes(x="Axis.1", y="Axis.2", fill=color, group=color),
                alpha=0.1,
                color=None,
            )

    p = p + geom_point(size=3)

    if label and label in plot_df.columns:
        # Nudge proportional to the y-axis range so labels clear the points
        # regardless of axis scale.
        y_range = (plot_df["Axis.2"].max() - plot_df["Axis.2"].min()) or 1.0
        p = p + geom_text(aes(label=label), nudge_y=0.02 * y_range, size=7)

    if title:
        p = p + labs(title=title)

    return p


def _convex_hull_df(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return a DataFrame of convex hull vertices per group for geom_polygon.

    Groups with fewer than 3 points cannot form a hull and are skipped.
    """
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    rows: list[dict] = []
    for group, sub in df.groupby(group_col, sort=False, observed=True):
        pts = sub[["Axis.1", "Axis.2"]].values
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            # vertices are already in CCW order; close the polygon
            for idx in np.append(hull.vertices, hull.vertices[0]):
                rows.append(
                    {"Axis.1": pts[idx, 0], "Axis.2": pts[idx, 1], group_col: group}
                )
        except QhullError:
            continue
    return pd.DataFrame(rows)


def _plot_scree(ord: Any, title: str | None = None) -> Any:
    """Scree plot of eigenvalues / proportion explained."""
    from plotnine import aes, geom_line, geom_point, ggplot, labs, xlab, ylab

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


def _split_df(ps: Phyloseq, ord: Any) -> pd.DataFrame:
    """Assemble the combined samples+taxa DataFrame used by the split plot."""
    frames = []
    sam_df = pd.DataFrame(
        ord.samples.values[:, :2],
        index=ord.samples.index,
        columns=["Axis.1", "Axis.2"],
    )
    sam_df["_panel"] = "Samples"
    if ps.sample_data is not None:
        sam_df = sam_df.join(ps.sample_data.to_frame(), how="left")
    frames.append(sam_df)

    if ord.features is not None:
        feat_xy = _rescale_biplot_scores(
            ord.samples.values[:, :2],
            ord.features.values[:, :2],
        )
        feat_df = pd.DataFrame(
            feat_xy,
            index=ord.features.index,
            columns=["Axis.1", "Axis.2"],
        )
        feat_df["_panel"] = "Taxa"
        if ps.tax_table is not None:
            feat_df = feat_df.join(ps.tax_table.to_frame(), how="left")
        frames.append(feat_df)

    return pd.concat(frames).reset_index()


def _plot_split(
    ps: Phyloseq,
    ord: Any,
    color: str | None = None,
    shape: str | None = None,
    label: str | None = None,
    title: str | None = None,
    show_hull: bool = False,
) -> Any:
    """Split biplot: samples and taxa in side-by-side facets."""
    from plotnine import (aes, facet_wrap, geom_point, geom_polygon,
                          geom_text, ggplot, labs)

    combined = _split_df(ps, ord)
    mapping: dict[str, str] = {"x": "Axis.1", "y": "Axis.2"}
    if color and color in combined.columns:
        mapping["color"] = color
    if shape and shape in combined.columns:
        mapping["shape"] = shape

    p = ggplot(combined, aes(**mapping))

    # Convex hull shading on the Samples panel only (opt-in).
    if show_hull and color and color in combined.columns:
        sam_only = combined[combined["_panel"] == "Samples"].dropna(subset=[color])
        hull_data = _convex_hull_df(sam_only, color)
        if not hull_data.empty:
            hull_data["_panel"] = "Samples"
            p = p + geom_polygon(
                data=hull_data,
                mapping=aes(x="Axis.1", y="Axis.2", fill=color, group=color),
                alpha=0.1,
                color=None,
            )

    p = p + geom_point(size=3) + facet_wrap("~_panel")

    if label and label in combined.columns:
        y_range = (combined["Axis.2"].max() - combined["Axis.2"].min()) or 1.0
        p = p + geom_text(aes(label=label), nudge_y=0.02 * y_range, size=7)

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    ps: Phyloseq,
    method: str = "NMDS",
    distance: str = "bray",
    trans: str | None = "log4",
    low: str = "#000033",
    high: str = "#66CCFF",
    na_value: str = "black",
    title: str | None = None,
) -> Any:
    """Abundance heatmap with samples *and* taxa reordered by ordination.

    Both axes are reordered using the ordination result (samples along x,
    taxa/OTUs along y), matching R phyloseq.  Zero/NA abundances are mapped
    to ``na_value`` rather than the gradient's low colour.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    method:
        Ordination method used to reorder samples and taxa.
    distance:
        Distance metric for ordination.
    trans:
        Transformation applied to abundances before plotting.
        ``"log4"`` (default) computes ``log4(x)`` with zeros kept as missing
        (so they map to ``na_value``); ``None`` uses raw counts.
    low, high:
        Gradient colour endpoints.
    na_value:
        Colour for zero/NA cells.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_heatmap(physeq, method, distance, trans, low, high,
                              na.value, title)
    """
    from plotnine import (aes, element_text, geom_tile, ggplot, labs,
                          scale_fill_gradient, theme)

    from pyloseq._manipulation import psmelt  # noqa: PLC0415
    from pyloseq._ordination import ordinate  # noqa: PLC0415

    # Ordinate to get sample AND taxa ordering.
    sample_order: list = list(ps.sample_names)
    taxa_order: list = list(ps.taxa_names)
    try:
        ord_result = ordinate(ps, method=method, distance=distance)
        first_axis = ord_result.samples.columns[0]
        sample_order = list(ord_result.samples.sort_values(first_axis).index)
        # Reorder taxa by their ordination scores when available.
        if getattr(ord_result, "features", None) is not None:
            feat_first = ord_result.features.columns[0]
            taxa_order = list(ord_result.features.sort_values(feat_first).index)
    except (ValueError, RuntimeError, KeyError, pyloseqValidationError) as e:
        warnings.warn(
            f"Ordination failed ({e!r}); using original sample/taxa order.",
            stacklevel=2,
        )

    long_df = psmelt(ps)

    # Apply transformation.  Zeros become NaN so they render as na_value
    # rather than being lumped into the gradient low colour.
    if trans == "log4":
        vals = long_df["Abundance"].astype(float)
        with np.errstate(divide="ignore"):
            transformed = np.log(vals.where(vals > 0)) / np.log(4)
        long_df["Abundance"] = transformed
    elif trans is not None:
        raise pyloseqValidationError(f"Unknown trans '{trans}'. Use 'log4' or None.")

    # Order both axes by the ordination.
    long_df["Sample"] = pd.Categorical(
        long_df["Sample"], categories=sample_order, ordered=True
    )
    long_df["OTU"] = pd.Categorical(
        long_df["OTU"], categories=taxa_order, ordered=True
    )

    p = (
        ggplot(long_df, aes(x="Sample", y="OTU", fill="Abundance"))
        + geom_tile()
        + scale_fill_gradient(low=low, high=high, na_value=na_value)
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
    kind: str = "samples",
    distance: str = "jaccard",
    max_dist: float = 0.4,
    keep_isolates: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a sample (or taxa) network based on a distance threshold.

    Edges are drawn between nodes whose distance is strictly less than
    ``max_dist`` (matching R phyloseq's ``max.dist`` semantics).

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    kind:
        ``"samples"`` (default) or ``"taxa"``.
    distance:
        Distance metric (see :func:`pyloseq.distance`).  For ``"jaccard"`` a
        binary (presence/absence) Jaccard distance is used, consistent with
        phyloseq's default network distance.
    max_dist:
        Maximum distance for an edge to be drawn (strict ``<``).
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

    if "type" in kwargs:
        warnings.warn(
            "The 'type' parameter is deprecated; use 'kind' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kind = kwargs.pop("type")

    from pyloseq._distances import distance as _distance  # noqa: PLC0415

    # Note: _distance handles presence/absence binarization internally for
    # metrics that require it (e.g. jaccard), so we do not pass a binary flag
    # here — scipy's pdist would reject it.
    dm = _distance(ps, distance, kind=kind)
    ids = list(dm.ids)
    data = np.array(dm.data)

    g = nx.Graph()
    g.add_nodes_from(ids)

    for i, u in enumerate(ids):
        for j, v in enumerate(ids):
            if j <= i:
                continue
            # Strict less-than, matching R's max.dist behaviour.
            if data[i, j] < max_dist:
                g.add_edge(u, v, weight=float(data[i, j]))

    if not keep_isolates:
        isolates = list(nx.isolates(g))
        g.remove_nodes_from(isolates)

    # Attach metadata as node attributes.
    if kind == "samples" and ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        for node in g.nodes:
            if node in sam_df.index:
                for col in sam_df.columns:
                    g.nodes[node][col] = sam_df.loc[node, col]
    elif kind == "taxa" and ps.tax_table is not None:
        tax_df = ps.tax_table.to_frame()
        for node in g.nodes:
            if node in tax_df.index:
                for col in tax_df.columns:
                    g.nodes[node][col] = tax_df.loc[node, col]

    return g


def plot_network(
    g: Any,
    ps: Phyloseq,
    color: str | None = None,
    shape: str | None = None,
    line_weight: float = 0.5,
    line_color: str = "grey",
    line_alpha: float = 0.4,
    point_size: float = 5.0,
    label: str | None = None,
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
    shape:
        Node attribute column for point shape.
    line_weight:
        Edge line width.
    line_color:
        Edge colour.
    line_alpha:
        Edge opacity.
    point_size:
        Node point size.
    label:
        Node attribute column (or ``"node"`` for the node id) to draw as text
        labels next to each node.
    layout:
        NetworkX layout algorithm name (e.g. ``"fruchterman_reingold"``).
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_network(ig, physeq, color, shape, line_weight,
                              line_color, line_alpha, point_size, label,
                              layout, title)
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "plot_network requires networkx. Install it with: pip install networkx"
        ) from e

    from plotnine import (aes, geom_point, geom_segment, geom_text, ggplot,
                          labs)

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
        pd.DataFrame(edge_rows)
        if edge_rows
        else pd.DataFrame(columns=["x", "y", "xend", "yend"])
    )

    mapping: dict[str, str] = {"x": "x", "y": "y"}
    if color and color in nodes_df.columns:
        mapping["color"] = color
    if shape and shape in nodes_df.columns:
        mapping["shape"] = shape

    p = ggplot(nodes_df, aes(**mapping))

    if not edges_df.empty:
        p = p + geom_segment(
            data=edges_df,
            mapping=aes(x="x", y="y", xend="xend", yend="yend"),
            color=line_color,
            size=line_weight,
            alpha=line_alpha,
        )

    p = p + geom_point(size=point_size)

    if label and label in nodes_df.columns:
        y_range = (nodes_df["y"].max() - nodes_df["y"].min()) or 1.0
        p = p + geom_text(
            aes(label=label),
            nudge_y=0.02 * y_range,
            size=7,
        )

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# plot_tree
# ---------------------------------------------------------------------------


def _tree_layout(
    tree: Any,
    ladderize: bool | str = False,
) -> tuple[pd.DataFrame, dict, dict, dict, dict, list]:
    """Compute (x, y) for every node and the tree edges as segments.

    x is cumulative branch length from the root; y is the leaf ordinal
    (mean of children's y for internal nodes).  Returns the segments
    DataFrame plus lookup tables used by :func:`plot_tree`.
    """
    if ladderize:
        direction = "right" if ladderize is True else ladderize
        for node in tree.postorder(include_self=True):
            if not node.is_tip():
                node.children.sort(
                    key=lambda c: sum(1 for _ in c.tips(include_self=True)),
                    reverse=(direction == "right"),
                )

    # Detect whether the tree has meaningful branch lengths
    has_lengths = any(
        n.length is not None and n.length > 0
        for n in tree.traverse(include_self=False)
    )

    tip_order: list[str] = [t.name for t in tree.tips(include_self=False)]
    tip_y: dict[str, float] = {name: i for i, name in enumerate(tip_order)}

    tip_x: dict[str, float] = {}
    node_x: dict[int, float] = {}
    node_y: dict[int, float] = {}

    def _set_x(node: Any, parent_x: float = 0.0) -> None:
        bl = node.length if node.length is not None else 0.0
        if not has_lengths and node.parent is not None:
            bl = 1.0
        x = parent_x + bl
        if node.is_tip():
            tip_x[node.name] = x
        else:
            node_x[id(node)] = x
        for child in node.children:
            _set_x(child, x)

    _set_x(tree)

    def _y_of(node: Any) -> float:
        return tip_y[node.name] if node.is_tip() else node_y[id(node)]

    def _set_y(node: Any) -> float:
        if node.is_tip():
            return tip_y[node.name]
        ys = [_set_y(c) for c in node.children]
        y = (min(ys) + max(ys)) / 2.0
        node_y[id(node)] = y
        return y

    _set_y(tree)

    def _x_of(node: Any) -> float:
        return tip_x[node.name] if node.is_tip() else node_x[id(node)]

    segs: list[dict[str, float]] = []
    # Horizontal segments from parent to each child
    for node in tree.traverse(include_self=False):
        x_p = _x_of(node.parent)
        x_c = _x_of(node)
        y_c = _y_of(node)
        segs.append({"x": x_p, "y": y_c, "xend": x_c, "yend": y_c})

    # Vertical connectors at each internal node spanning its children
    for node in tree.traverse(include_self=True):
        if not node.is_tip() and node.children:
            x_n = node_x[id(node)]
            child_ys = [_y_of(c) for c in node.children]
            segs.append(
                {"x": x_n, "y": min(child_ys), "xend": x_n, "yend": max(child_ys)}
            )

    return pd.DataFrame(segs), tip_x, tip_y, node_x, node_y, tip_order


def plot_tree(
    ps: Phyloseq,
    method: str = "sampledodge",
    color: str | None = None,
    shape: str | None = None,
    size: str | None = None,
    label_tips: str | None = None,
    text_size: float = 5.0,
    sizebase: float = 5.0,
    base_spacing: float = 0.02,
    min_abundance: float = 0.0,
    ladderize: bool | str = False,
    title: str | None = None,
) -> Any:
    """Phylogenetic tree with per-sample points at the tips.

    Parameters
    ----------
    ps:
        ``Phyloseq`` object; must include a phylogenetic tree (``phy_tree``).
    method:
        ``"sampledodge"`` (default) draws one point per sample at each tip,
        offset rightward along x.  ``"treeonly"`` draws the tree alone.
    color:
        Sample (sampledodge) or tax_table (taxa) column for point colour.
    shape:
        Sample metadata column for point shape.
    size:
        ``"Abundance"`` to scale point size by abundance (the size legend
        reports the original abundance breaks, not the log-transformed
        values), any numeric metadata column, or ``None`` for fixed size.
    label_tips:
        ``tax_table`` column whose values label each tip (e.g. ``"Genus"``).
    text_size:
        Font size for tip labels.
    sizebase:
        Log base for abundance-to-size transform.
    base_spacing:
        Fractional x-spacing between dodged points, as a fraction of the tree
        x-range.
    min_abundance:
        Drop sample points whose abundance is less than or equal to this
        value (sampledodge only).
    ladderize:
        ``False``, ``True``/``"right"``, or ``"left"``.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    R reference: plot_tree(physeq, method, color, shape, size, label.tips,
                           text.size, sizebase, base.spacing, min.abundance,
                           ladderize, title)
    """
    from plotnine import (aes, element_blank, geom_point, geom_segment,
                          geom_text, ggplot, labs, scale_size_continuous,
                          theme, theme_minimal)

    if getattr(ps, "phy_tree", None) is None:
        raise pyloseqValidationError(
            "plot_tree requires a phylogenetic tree on the Phyloseq object."
        )
    if method not in ("sampledodge", "treeonly"):
        raise pyloseqValidationError(
            f"Unknown plot_tree method '{method}'. Use 'sampledodge' or 'treeonly'."
        )

    segs_df, tip_x, tip_y, node_x, _node_y, tip_order = _tree_layout(
        ps.phy_tree._tree, ladderize=ladderize
    )

    all_x = list(tip_x.values()) + list(node_x.values())
    x_range = (max(all_x) - min(all_x)) or 1.0

    p = ggplot() + geom_segment(
        data=segs_df, mapping=aes(x="x", y="y", xend="xend", yend="yend")
    )

    long_df: pd.DataFrame | None = None
    last_offset = 0  # used to position tip labels past the dot column

    if method == "sampledodge":
        from pyloseq._manipulation import psmelt  # noqa: PLC0415

        long_df = psmelt(ps)
        # Drop points at or below the threshold (<=), matching the documented
        # behaviour.
        long_df = long_df[long_df["Abundance"] > min_abundance].copy()
        long_df = long_df[long_df["OTU"].isin(tip_x)]

        sample_order = sorted(long_df["Sample"].unique())
        offset = {s: i + 1 for i, s in enumerate(sample_order)}
        last_offset = len(sample_order)

        long_df["x_point"] = (
            long_df["OTU"].map(tip_x)
            + long_df["Sample"].map(offset) * base_spacing * x_range
        )
        long_df["y_point"] = long_df["OTU"].map(tip_y)

        mapping: dict[str, str] = {"x": "x_point", "y": "y_point"}
        if color and color in long_df.columns:
            mapping["color"] = color
        if shape and shape in long_df.columns:
            mapping["shape"] = shape

        size_breaks = None
        size_labels = None
        if size == "Abundance":
            # Transform for the visual radius, but compute legend breaks in
            # the original abundance units so the legend is meaningful.
            ab = long_df["Abundance"].clip(lower=1.0)
            long_df["_size"] = np.log(ab) / np.log(sizebase)
            mapping["size"] = "_size"

            raw_min = float(long_df["Abundance"].min())
            raw_max = float(long_df["Abundance"].max())
            raw_breaks = np.unique(
                np.linspace(max(raw_min, 1.0), max(raw_max, 1.0), num=4).round()
            )
            size_breaks = list(np.log(np.clip(raw_breaks, 1.0, None)) / np.log(sizebase))
            size_labels = [f"{int(b)}" for b in raw_breaks]
        elif size and size in long_df.columns:
            mapping["size"] = size

        p = p + geom_point(data=long_df, mapping=aes(**mapping))
        if size == "Abundance":
            p = p + scale_size_continuous(
                name="Abundance",
                breaks=size_breaks,
                labels=size_labels,
            )

    # Tip labels
    if label_tips:
        if ps.tax_table is None:
            raise pyloseqValidationError(
                "label_tips was provided but Phyloseq has no tax_table."
            )
        tax_df = ps.tax_table.to_frame()
        if label_tips not in tax_df.columns:
            raise pyloseqValidationError(
                f"label_tips column '{label_tips}' not found in tax_table."
            )
        label_x_offset = (last_offset + 1) * base_spacing * x_range
        tip_df = pd.DataFrame(
            {
                "OTU": tip_order,
                "x_label": [tip_x[t] + label_x_offset for t in tip_order],
                "y_tip": [tip_y[t] for t in tip_order],
                "_label": [
                    tax_df.loc[t, label_tips] if t in tax_df.index else ""
                    for t in tip_order
                ],
            }
        )
        p = p + geom_text(
            data=tip_df,
            mapping=aes(x="x_label", y="y_tip", label="_label"),
            ha="left",
            size=text_size,
        )

    p = (
        p
        + theme_minimal()
        + theme(
            axis_text_y=element_blank(),
            axis_ticks_major_y=element_blank(),
            panel_grid=element_blank(),
        )
    )
    if title:
        p = p + labs(title=title)

    return p