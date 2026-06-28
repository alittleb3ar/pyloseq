"""Plotting functions mirroring R phyloseq's visualization API.

All functions return ``plotnine.ggplot`` objects (except :func:`make_network`
which returns a ``networkx.Graph``).  The underlying data is always available
via the plot's ``.data`` attribute.

"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import pandas as pd
import plotnine as _pn
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_errorbar,
    geom_line,
    geom_point,
    geom_polygon,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_shape_manual,
    scale_size_continuous,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_discrete,
    theme,
    theme_minimal,
    xlab,
    ylab,
)
from plotnine import facet_grid as pg_facet_grid
from scipy.spatial import ConvexHull, QhullError

from pyloseq._distances import distance as _distance
from pyloseq._diversity import _ALL_MEASURES, estimate_richness
from pyloseq._exceptions import pyloseqValidationError
from pyloseq._manipulation import psmelt
from pyloseq._ordination import ordinate

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

    R reference: plot_bar(physeq, x, y, fill, facet_grid, title)

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

    """

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
    boxplot: bool = True,
) -> Any:
    """Alpha diversity box-and-point plots, faceted by measure.

    Standard-error whiskers are drawn for any measure that has a matching
    ``se.<measure>`` column from :func:`estimate_richness` (e.g. ``se.chao1``,
    ``se.ACE``), matching R phyloseq.

    R reference: plot_richness(physeq, x, color, measures, title)

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
    boxplot:
        If ``True`` (default), layer a box-and-whisker summary under the
        points.  Set ``False`` for points only (e.g. when ``x`` groups few
        samples and the boxes add noise).

    Returns
    -------
    plotnine.ggplot

    """
    rich_df = estimate_richness(ps, measures=measures)

    # Separate the value columns from their standard-error partners.
    se_cols = [c for c in rich_df.columns if c.lower().startswith("se.")]
    se_map = {c[3:]: c for c in se_cols}  # measure name -> se column

    requested = measures or _ALL_MEASURES
    measure_cols = [c for c in rich_df.columns if c not in se_cols and c in requested]

    # Join sample metadata
    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        rich_df = rich_df.join(sam_df, how="left")

    # Use a collision-proof name for the sample id rather than the literal
    # string "index", which can clash with a metadata column.
    sample_id_col = "_sample_id"
    rich_df[sample_id_col] = rich_df.index

    x_col = sample_id_col if x == "samples" else x

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

    # Box-and-whisker *or* points, not both: the box summarises the same
    # per-sample distribution the points would show, so layering both is
    # redundant.  boxplot=True draws the box; boxplot=False draws the points.
    p = p + geom_boxplot(alpha=0.5) if boxplot else p + geom_point(size=2)

    # Error bars where SE is available.
    if long["SE"].notna().any():
        p = p + geom_errorbar(
            aes(ymin="ymin", ymax="ymax"),
            width=0.2,
        )

    p = (
        p
        + facet_wrap("~Measure", scales="free_y")
        + theme(axis_text_x=element_text(rotation=90, hjust=1))
    )

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# plot_rarefaction_curve
# ---------------------------------------------------------------------------


def plot_rarefaction_curve(
    ps: Phyloseq,
    step: int = 500,
    n_steps: int = 30,
    color: str | None = None,
    rng_seed: int | None = None,
    title: str | None = None,
) -> Any:
    """Rarefaction curves showing observed richness vs. sequencing depth.

    For each sample a curve is drawn by randomly subsampling (without
    replacement) the reads at ``n_steps`` depths between ``step`` and the
    minimum sample depth, then counting the number of distinct observed taxa
    at each depth.  The minimum depth sets the right-hand end of all curves
    so that every sample reaches the same maximum depth point.

    R reference: vegan::rarecurve(t(otu_table(physeq)), step=...) or
                 microbiome::plot_richness_estimates (depth-based variant)

    Parameters
    ----------
    ps:
        ``Phyloseq`` object. OTU table values must be integer counts.
    step:
        Starting depth for the rarefaction grid (first subsampling depth).
    n_steps:
        Number of evenly-spaced depth points from ``step`` to the minimum
        sample depth.
    color:
        Column in ``sample_data`` to color the curves by.  If ``None`` all
        curves use the default color.
    rng_seed:
        Seed for the subsampling RNG.  Pass ``None`` for non-reproducible
        draws.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot
        The underlying ``data`` attribute contains columns ``Sample``,
        ``Depth``, ``Observed``, and any column named by ``color``.
    """
    from pyloseq._manipulation import _otu_taxa_rows  # noqa: PLC0415

    rng = np.random.default_rng(rng_seed)

    otu_taxa = _otu_taxa_rows(ps)  # taxa × samples
    sample_sums = otu_taxa.sum(axis=0)
    min_depth = int(sample_sums.min())

    if min_depth < step:
        raise pyloseqValidationError(
            f"Minimum sample depth ({min_depth}) is less than step ({step}). "
            "Lower step or rarefy to a higher minimum depth."
        )

    depths = np.linspace(step, min_depth, n_steps, dtype=int)

    rows: list[dict[str, Any]] = []
    for sample in ps.sample_names:
        counts = otu_taxa[sample].values.astype(int)
        total = int(counts.sum())
        pool = np.repeat(np.arange(len(counts)), counts)
        for d in depths:
            if d > total:
                break
            drawn = rng.choice(pool, size=d, replace=False)
            rows.append({"Sample": sample, "Depth": int(d), "Observed": int(np.unique(drawn).size)})

    curve_df = pd.DataFrame(rows)

    if color and ps.sample_data is not None and color in ps.sample_data.to_frame().columns:
        sam_df = ps.sample_data.to_frame()[[color]]
        curve_df = curve_df.join(sam_df, on="Sample")

    mapping: dict[str, str] = {"x": "Depth", "y": "Observed", "group": "Sample"}
    if color and color in curve_df.columns:
        mapping["color"] = color

    p = (
        ggplot(curve_df, aes(**mapping))
        + geom_line(alpha=0.8)
        + xlab("Sequencing depth (reads)")
        + ylab("Observed taxa")
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
    samp_span: float = np.nanmax(np.abs(sample_xy)) or 1.0
    feat_span: float = np.nanmax(np.abs(feature_xy)) or 1.0
    return cast(np.ndarray, feature_xy * (samp_span / feat_span))


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

    R reference: plot_ordination(physeq, ordination, type, color, shape,
                                 label, title, justDF)

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


    """
    if "type" in kwargs:
        warnings.warn(
            "The 'type' parameter is deprecated; use 'kind' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kind = kwargs.pop("type")

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
            ps,
            ord,
            color=color,
            shape=shape,
            label=label,
            title=title,
            show_hull=show_hull,
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

    p = ggplot(plot_df, aes(**mapping)) + xlab(_axis_label(1)) + ylab(_axis_label(2))

    # Convex hull shading per group (opt-in; samples kind only since biplot
    # mixes taxa rows into the frame).
    if show_hull and color and color in plot_df.columns and kind == "samples":
        p = _add_hull_layer(p, plot_df, color)

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
    rows: list[dict[str, Any]] = []
    for group, sub in df.groupby(group_col, sort=False, observed=True):
        pts = sub[["Axis.1", "Axis.2"]].values
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            # vertices are already in CCW order; close the polygon
            idx: int
            for idx in np.append(hull.vertices, hull.vertices[0]):
                rows.append(
                    {"Axis.1": pts[idx, 0], "Axis.2": pts[idx, 1], group_col: group}
                )
        except QhullError:
            continue
    return pd.DataFrame(rows)


def _add_hull_layer(
    p: Any,
    df: pd.DataFrame,
    color_col: str,
    panel_col: str | None = None,
    panel_val: str | None = None,
) -> Any:
    """Append a convex hull ``geom_polygon`` layer to *p* if the data allow it.

    Parameters
    ----------
    p:
        A ``plotnine.ggplot`` object to augment.
    df:
        DataFrame containing at least ``Axis.1``, ``Axis.2``, and *color_col*.
    color_col:
        Column used for group colouring.
    panel_col:
        Optional facet column; when provided only rows where ``panel_col ==
        panel_val`` are passed to hull computation, and the column is set on
        the hull data so ``facet_wrap`` places it correctly.
    panel_val:
        Required when *panel_col* is given.
    """
    sub = df.dropna(subset=[color_col])
    if panel_col is not None and panel_val is not None:
        sub = sub[sub[panel_col] == panel_val]
    hull_data = _convex_hull_df(sub, color_col)
    if hull_data.empty:
        return p
    if panel_col is not None and panel_val is not None:
        hull_data[panel_col] = panel_val
    return p + geom_polygon(
        data=hull_data,
        mapping=aes(x="Axis.1", y="Axis.2", fill=color_col, group=color_col),
        alpha=0.1,
        color=None,
    )


def _plot_scree(ord: Any, title: str | None = None) -> Any:
    """Scree plot of eigenvalues / proportion explained."""
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
    combined = _split_df(ps, ord)
    mapping: dict[str, str] = {"x": "Axis.1", "y": "Axis.2"}
    if color and color in combined.columns:
        mapping["color"] = color
    if shape and shape in combined.columns:
        mapping["shape"] = shape

    p = ggplot(combined, aes(**mapping))

    # Convex hull shading on the Samples panel only (opt-in).
    if show_hull and color and color in combined.columns:
        p = _add_hull_layer(p, combined, color, panel_col="_panel", panel_val="Samples")

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
    method: str | None = "NMDS",
    distance: str = "bray",
    trans: str | None = "log4",
    low: str = "#000033",
    high: str = "#66CCFF",
    na_value: str = "black",
    title: str | None = None,
    label: str | None = None,
    taxa_label: str | None = None,
) -> Any:
    """Abundance heatmap with samples *and* taxa reordered by ordination.

    Both axes are reordered using the ordination result (samples along x,
    taxa/OTUs along y), matching R phyloseq.  Zero/NA abundances are mapped
    to ``na_value`` rather than the gradient's low colour.

    R reference: plot_heatmap(physeq, method, distance, trans, low, high,
                              na.value, title, sample.label, taxa.label)

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    method:
        Ordination method used to reorder samples and taxa.  Pass ``None``
        to skip ordination and preserve the original sample/taxa order.
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
    label:
        Column in ``sample_data`` whose values label the x-axis ticks
        instead of sample names.  Samples are still ordered by ordination
        (or original order when ``method=None``); only the tick text
        changes.  A warning is emitted if the column is not found.
        Mirrors R phyloseq's ``sample.label``.
    taxa_label:
        Taxonomic rank (e.g. ``"Class"``) whose values label the y-axis
        ticks instead of OTU/taxa names.  Taxa are still ordered by
        ordination (or original order when ``method=None``); only the tick
        text changes.  A warning is emitted if the rank is not found.
        Mirrors R phyloseq's ``taxa.label``.

    Returns
    -------
    plotnine.ggplot

    """
    # Ordinate to get sample AND taxa ordering.
    sample_order: list[str] = list(ps.sample_names)
    taxa_order: list[str] = list(ps.taxa_names)
    if method is not None:
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
    long_df["OTU"] = pd.Categorical(long_df["OTU"], categories=taxa_order, ordered=True)

    p = (
        ggplot(long_df, aes(x="Sample", y="OTU", fill="Abundance"))
        + geom_tile()
        + scale_fill_gradient(low=low, high=high, na_value=na_value)
        + theme(axis_text_x=element_text(rotation=90, hjust=1))
    )

    if label is not None:
        if ps.sample_data is None or label not in long_df.columns:
            warnings.warn(
                f"label '{label}' not found in sample_data; using sample names.",
                stacklevel=2,
            )
        else:
            labels_dict: dict[str, str] = {
                str(k): v
                for k, v in long_df[["Sample", label]]
                .drop_duplicates()
                .set_index("Sample")[label]
                .astype(str)
                .to_dict()
                .items()
            }
            p = p + scale_x_discrete(labels=labels_dict)

    if taxa_label is not None:
        if ps.tax_table is None or taxa_label not in long_df.columns:
            warnings.warn(
                f"taxa_label '{taxa_label}' not found in tax_table; "
                "using taxa names.",
                stacklevel=2,
            )
        else:
            taxa_labels_dict: dict[str, str] = {
                str(k): v
                for k, v in long_df[["OTU", taxa_label]]
                .drop_duplicates("OTU")
                .set_index("OTU")[taxa_label]
                .astype(str)
                .to_dict()
                .items()
            }
            p = p + scale_y_discrete(labels=taxa_labels_dict)

    if title:
        p = p + labs(title=title)

    return p


# ---------------------------------------------------------------------------
# make_network / plot_network
# ---------------------------------------------------------------------------


def make_network(
    ps: Phyloseq,
    kind: str = "samples",
    distance: str | Any = "jaccard",
    max_dist: float = 0.4,
    keep_isolates: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a sample (or taxa) network based on a distance threshold.

    Edges are drawn between nodes whose distance is strictly less than
    ``max_dist`` (matching R phyloseq's ``max.dist`` semantics).

    R reference: make_network(physeq, type, distance, max.dist, keep.isolates)
    R reference: plot_net(physeq, distance=as.dist(...), ...)  — precomputed DM

    Parameters
    ----------
    ps:
        ``Phyloseq`` object.
    kind:
        ``"samples"`` (default) or ``"taxa"``.
    distance:
        Distance metric string (see :func:`pyloseq.distance`) **or** a
        precomputed ``skbio.stats.distance.DistanceMatrix``.  Passing a
        ``DistanceMatrix`` mirrors R's ``plot_net(distance=as.dist(...))``
        pattern (e.g. from :func:`pyloseq.gunifrac`).
    max_dist:
        Maximum distance for an edge to be drawn (strict ``<``).
    keep_isolates:
        If ``False`` (default), remove nodes with no edges.

    Returns
    -------
    networkx.Graph

    """
    from skbio.stats.distance import DistanceMatrix as _DM

    if "type" in kwargs:
        warnings.warn(
            "The 'type' parameter is deprecated; use 'kind' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kind = kwargs.pop("type")

    # Note: _distance handles presence/absence binarization internally for
    # metrics that require it (e.g. jaccard), so we do not pass a binary
    # flag here — scipy's pdist would reject it.
    dm = distance if isinstance(distance, _DM) else _distance(ps, distance, kind=kind)
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

    R reference: plot_network(ig, physeq, color, shape, line_weight,
                              line_color, line_alpha, point_size, label,
                              layout, title)

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

    """
    layout_fn = getattr(nx, f"{layout}_layout", nx.spring_layout)
    pos = layout_fn(g)

    # Nodes data frame
    rows = []
    for node, (x, y) in pos.items():
        row: dict[str, Any] = {"node": node, "x": x, "y": y}
        row.update(g.nodes[node])
        rows.append(row)
    nodes_df = pd.DataFrame(rows)

    # Edges data frame — include stored distance as "distance" column so that
    # edge width can be mapped to it (closer samples → thicker line, matching R).
    edge_rows = []
    for u, v, attrs in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_rows.append(
            {
                "x": x0,
                "y": y0,
                "xend": x1,
                "yend": y1,
                "distance": attrs.get("weight", 0.0),
            }
        )
    edges_df = (
        pd.DataFrame(edge_rows)
        if edge_rows
        else pd.DataFrame(columns=["x", "y", "xend", "yend", "distance"])
    )

    # Only x/y go in the global aes so that geom_segment (which uses its own
    # data=edges_df) never inherits color/shape mappings that don't exist in
    # edges_df and would cause a plotnine evaluation error.
    point_mapping: dict[str, str] = {}
    if color and color in nodes_df.columns:
        point_mapping["color"] = color
    if shape and shape in nodes_df.columns:
        point_mapping["shape"] = shape

    p = ggplot(nodes_df, aes(x="x", y="y"))

    if not edges_df.empty:
        # Map size to distance with an inverted range so that edges between
        # more-similar samples (smaller distance) appear thicker — matching R's
        # plot_net which uses scale_size_continuous(range=c(3, 0.5)).
        p = p + geom_segment(
            data=edges_df,
            mapping=aes(x="x", y="y", xend="xend", yend="yend", size="distance"),
            color=line_color,
            alpha=line_alpha,
        )
        p = p + scale_size_continuous(range=(3, 0.5), name="Distance")

    pt_aes = aes(**point_mapping) if point_mapping else None
    p = p + geom_point(mapping=pt_aes, size=point_size)

    # plotnine's default discrete shape palette has 6 entries. When the shape
    # column has more unique values (e.g. 22 latrines), guide_legend.draw()
    # crashes with an IndexError even with a custom scale. Suppress the shape
    # legend in that case (shapes still differ in the plot; legend is
    # unreadable at that scale anyway) and warn the caller.
    _SHAPE_MARKERS = [
        "o",
        "s",
        "^",
        "v",
        "<",
        ">",
        "D",
        "p",
        "*",
        "h",
        "H",
        "d",
        "P",
        "X",
        "8",
        "1",
        "2",
        "3",
        "4",
        "+",
        "x",
        "|",
        "_",
    ]
    if shape and shape in nodes_df.columns:
        n_unique = nodes_df[shape].nunique()
        if n_unique > 6:
            warnings.warn(
                f"plot_network: shape='{shape}' has {n_unique} unique values "
                f"(> 6). Shapes are still drawn distinctly but the shape legend "
                f"is suppressed to avoid a plotnine rendering crash.",
                stacklevel=2,
            )
            p = p + scale_shape_manual(
                values=_SHAPE_MARKERS[:n_unique],
                breaks=[],  # empty breaks → no legend keys → avoids plotnine crash
            )

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


def _auto_text_size(n_tips: int) -> float:
    """Scale text size based on tip count, approximating R's manytextsize."""
    return float(max(1.5, min(6.0, 25.0 / max(n_tips, 1))))


def _tree_layout(
    tree: Any,
    ladderize: bool | str = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
    dict[str, float],
    dict[int, float],
    dict[int, float],
    list[str],
]:
    """Compute tree geometry and return separate edge and vertical-connector DataFrames.

    Returns
    -------
    edge_df:
        One row per tree edge (parent → child).
        Columns: ``xleft`` (parent x), ``xright`` (child x), ``y`` (child y),
        ``OTU`` (tip name for leaf edges, ``None`` for internal edges).
    vert_df:
        One row per internal node.
        Columns: ``x``, ``vmin``, ``vmax`` (y-extent of children).
    tip_x, tip_y, node_x, node_y:
        Coordinate lookup dicts.
    tip_order:
        Tip names in display order (bottom → top).
    """
    if ladderize:
        direction = "right" if ladderize is True else ladderize
        for node in tree.postorder(include_self=True):
            if not node.is_tip():
                node.children.sort(
                    key=lambda c: sum(1 for _ in c.tips()),
                    reverse=(direction == "right"),
                )

    has_lengths = any(
        n.length is not None and n.length > 0 for n in tree.traverse(include_self=False)
    )

    tip_order: list[str] = [t.name for t in tree.tips()]
    tip_y: dict[str, float] = {name: float(i) for i, name in enumerate(tip_order)}
    tip_x: dict[str, float] = {}
    node_x: dict[int, float] = {}
    node_y: dict[int, float] = {}

    def _set_x(node: Any, parent_x: float = 0.0) -> None:
        bl = (
            (node.length or 0.0)
            if has_lengths
            else (1.0 if node.parent is not None else 0.0)
        )
        x = parent_x + bl
        if node.is_tip():
            tip_x[node.name] = x
        else:
            node_x[id(node)] = x
        for child in node.children:
            _set_x(child, x)

    _set_x(tree)

    def _xof(node: Any) -> float:
        return tip_x[node.name] if node.is_tip() else node_x[id(node)]

    def _set_y(node: Any) -> float:
        if node.is_tip():
            return tip_y[node.name]
        ys = [_set_y(c) for c in node.children]
        y = (min(ys) + max(ys)) / 2.0
        node_y[id(node)] = y
        return y

    _set_y(tree)

    def _yof(node: Any) -> float:
        return tip_y[node.name] if node.is_tip() else node_y[id(node)]

    edges = [
        {
            "xleft": _xof(node.parent),
            "xright": _xof(node),
            "y": _yof(node),
            "OTU": node.name if node.is_tip() else None,
        }
        for node in tree.traverse(include_self=False)
    ]

    verts = [
        {
            "x": _xof(node),
            "vmin": min(_yof(c) for c in node.children),
            "vmax": max(_yof(c) for c in node.children),
        }
        for node in tree.traverse(include_self=True)
        if not node.is_tip() and node.children
    ]

    return (
        pd.DataFrame(edges),
        pd.DataFrame(verts),
        tip_x,
        tip_y,
        node_x,
        node_y,
        tip_order,
    )


def plot_tree(
    ps: Phyloseq,
    method: str = "sampledodge",
    color: str | None = None,
    shape: str | None = None,
    size: str | None = None,
    label_tips: str | None = None,
    text_size: float | None = None,
    sizebase: float = 5.0,
    base_spacing: float = 0.02,
    min_abundance: float = float("inf"),
    ladderize: bool | str = False,
    justify: str = "jagged",
    plot_margin: float = 0.2,
    figure_size: tuple[float, float] | None = None,
    title: str | None = None,
) -> Any:
    """Phylogenetic tree with per-sample points at the tips.

    R reference: plot_tree(physeq, method, color, shape, size, label.tips,
                           text.size, sizebase, base.spacing, min.abundance,
                           ladderize, justify, plot.margin, title)

    Parameters
    ----------
    ps:
        ``Phyloseq`` object; must include a phylogenetic tree (``phy_tree``).
    method:
        ``"sampledodge"`` (default) draws one point per sample at each tip,
        offset rightward along x.  ``"treeonly"`` draws the tree alone.
    color:
        Sample metadata or ``tax_table`` column for point colour.
    shape:
        Sample metadata column for point shape.
    size:
        ``"Abundance"`` to scale point size by log-transformed abundance,
        any numeric metadata column, or ``None`` for fixed size.
    label_tips:
        ``tax_table`` column whose values label each tip (e.g. ``"Genus"``).
    text_size:
        Font size for tip labels.  Auto-scaled from tip count when ``None``.
    sizebase:
        Log base for the abundance → size transform.
    base_spacing:
        Fractional x-step between dodged sample points, as a proportion of
        the maximum tip x value.
    min_abundance:
        Abundance threshold for printing per-point text labels.  Default
        ``inf`` suppresses all labels (matching R phyloseq).  Points
        themselves are always shown for ``Abundance > 0``.
    ladderize:
        ``False``, ``True`` / ``"right"`` (most-speciose clade at top), or
        ``"left"`` (most-speciose clade at bottom).
    justify:
        ``"jagged"`` (default) starts each tip's dodge column from its own x
        position.  ``"left"`` aligns all dodge columns at the rightmost tip.
    plot_margin:
        Fractional right-margin added beyond the last dodged point so that
        tip labels are not clipped.
    figure_size:
        ``(width, height)`` in inches.  When ``None`` (default), height is
        auto-scaled from the tip count (``0.2 * n_tips``, min 6) and width
        is fixed at 12.
    title:
        Plot title.

    Returns
    -------
    plotnine.ggplot

    """
    if ps.phy_tree is None:
        raise pyloseqValidationError(
            "plot_tree requires a phylogenetic tree on the Phyloseq object."
        )
    if method not in ("sampledodge", "treeonly"):
        raise pyloseqValidationError(
            f"Unknown plot_tree method '{method}'. Use 'sampledodge' or 'treeonly'."
        )
    if justify not in ("jagged", "left"):
        raise pyloseqValidationError(
            f"Unknown justify '{justify}'. Use 'jagged' or 'left'."
        )

    edge_df, vert_df, tip_x, tip_y, _node_x, _node_y, tip_order = _tree_layout(
        ps.phy_tree._tree, ladderize=ladderize
    )

    if figure_size is None:
        figure_size = (12.0, max(6.0, 0.2 * len(tip_order)))

    if text_size is None:
        # Scale to fill each tip's row: (figure_height × 72 pt/in) / n_tips,
        # halved for a comfortable line height, clamped to [3, 12] pt.
        points_per_tip = (figure_size[1] * 72.0) / max(len(tip_order), 1)
        text_size = float(max(3.0, min(12.0, points_per_tip / 2.0)))

    # plotnine refuses to draw figures larger than 25 inches by default.
    # Trees with many tips routinely need more height, so lift the limit.
    if max(figure_size) > 25:
        _pn.options.limitsize = False

    max_tip_x: float = max(tip_x.values()) if tip_x else 1.0

    # Base tree: horizontal edges then vertical connectors (matching R layer order)
    p = (
        ggplot()
        + geom_segment(
            data=edge_df, mapping=aes(x="xleft", xend="xright", y="y", yend="y")
        )
        + geom_segment(
            data=vert_df, mapping=aes(x="x", xend="x", y="vmin", yend="vmax")
        )
    )

    # ------------------------------------------------------------------ #
    # treeonly: optional tip labels, then return                          #
    # ------------------------------------------------------------------ #
    if method == "treeonly":
        if label_tips:
            _tip_labels_layer(
                p,
                ps,
                label_tips,
                tip_order,
                tip_x,
                tip_y,
                x_offset=0.0,
                text_size=text_size,
                color=color,
            )
        p = _apply_tree_theme(p, title, figure_size)
        if plot_margin > 0:
            p = p + scale_x_continuous(limits=(-0.01, max_tip_x * (1.0 + plot_margin)))
        return p

    # ------------------------------------------------------------------ #
    # sampledodge                                                         #
    # ------------------------------------------------------------------ #
    long_df = psmelt(ps)
    long_df = long_df[long_df["OTU"].isin(tip_x)].copy()

    # Jagged: strip zeros before assigning positions (R behaviour)
    if justify == "jagged":
        long_df = long_df[long_df["Abundance"] > 0].copy()

    # Fill NaN in discrete aesthetic columns so plotnine doesn't drop rows or
    # crash the legend when a tax_table rank is unassigned for some taxa.
    for _col in [color, shape]:
        if _col and _col in long_df.columns and long_df[_col].isna().any():
            long_df[_col] = long_df[_col].fillna("Unknown")

    # Dodge sort order: key columns then sample name (mirrors R setkeyv)
    sort_keys = ["OTU"] + [
        k for k in [color, shape, size] if k and k in long_df.columns
    ]
    if len(sort_keys) == 1:
        sort_keys.append("Sample")
    long_df = long_df.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    # Per-OTU 1-based dodge index
    long_df["_h_adj"] = long_df.groupby("OTU", sort=False, observed=True).cumcount() + 1

    if justify == "jagged":
        long_df["xdodge"] = (
            long_df["OTU"].map(tip_x) + long_df["_h_adj"] * base_spacing * max_tip_x
        )
    else:
        long_df["xdodge"] = max_tip_x + long_df["_h_adj"] * base_spacing * max_tip_x
        # Left-justify: strip zeros AFTER position assignment (R behaviour)
        long_df = long_df[long_df["Abundance"] > 0].copy()

    long_df["_y"] = long_df["OTU"].map(tip_y)

    # Point mapping: R maps both color and fill to the same variable
    pt_map: dict[str, str] = {"x": "xdodge", "y": "_y"}
    if color and color in long_df.columns:
        pt_map["color"] = color
        pt_map["fill"] = color
    if shape and shape in long_df.columns:
        pt_map["shape"] = shape

    if size == "Abundance":
        ab = long_df["Abundance"].clip(lower=1.0)
        long_df["_size"] = np.log(ab) / np.log(sizebase)
        pt_map["size"] = "_size"
        raw_min = float(long_df["Abundance"].min())
        raw_max = float(long_df["Abundance"].max())
        raw_breaks = np.unique(
            np.linspace(max(raw_min, 1.0), max(raw_max, 1.0), num=4).round()
        )
        p = p + geom_point(data=long_df, mapping=aes(**pt_map))
        p = p + scale_size_continuous(
            name="Abundance",
            breaks=list(np.log(raw_breaks.clip(1.0)) / np.log(sizebase)),
            labels=[f"{int(b)}" for b in raw_breaks],
        )
    elif size and size in long_df.columns:
        pt_map["size"] = size
        p = p + geom_point(data=long_df, mapping=aes(**pt_map))
    else:
        p = p + geom_point(data=long_df, mapping=aes(**pt_map))

    # Abundance text labels (only where Abundance >= min_abundance)
    if long_df["Abundance"].ge(min_abundance).any():
        lab_df = long_df[long_df["Abundance"] >= min_abundance]
        p = p + geom_text(
            data=lab_df,
            mapping=aes(x="xdodge", y="_y", label="Abundance"),
            size=text_size,
        )

    # Tip labels: anchored at the rightmost dodged point for each tip
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
        max_xdodge_per_otu: pd.Series = long_df.groupby("OTU")["xdodge"].max()
        # Null out color if it is a sample variable (R behaviour: per-tip labels
        # should not be colored by a per-sample variable)
        label_color = color
        if color and ps.sample_data is not None and color in ps.sample_variables:
            label_color = None
        tip_lab_df = pd.DataFrame(
            {
                "OTU": tip_order,
                "x_lab": [max_xdodge_per_otu.get(t, tip_x[t]) for t in tip_order],
                "_y": [tip_y[t] for t in tip_order],
                "_label": [
                    tax_df.loc[t, label_tips] if t in tax_df.index else ""
                    for t in tip_order
                ],
            }
        )
        lab_map: dict[str, str] = {"x": "x_lab", "y": "_y", "label": "_label"}
        if label_color and label_color in tip_lab_df.columns:
            lab_map["color"] = label_color
        p = p + geom_text(
            data=tip_lab_df,
            mapping=aes(**lab_map),
            nudge_x=base_spacing * max_tip_x,
            ha="left",
            size=text_size,
        )

    # X-axis limits: start from the farthest dodged point, extend by the label
    # nudge so text start positions are inside the plot, then add plot_margin
    # for breathing room (text itself extends rightward from the anchor).
    max_x = max(max_tip_x, float(long_df["xdodge"].max()))
    if label_tips:
        max_x += base_spacing * max_tip_x  # include the nudge offset
    max_x *= 1.0 + plot_margin
    p = p + scale_x_continuous(limits=(-0.01, max_x))

    return _apply_tree_theme(p, title, figure_size)


def _apply_tree_theme(
    p: Any,
    title: str | None,
    figure_size: tuple[float, float] | None = None,
) -> Any:
    """Attach the standard minimal tree theme and optional title."""
    p = (
        p
        + theme_minimal()
        + theme(
            axis_text_y=element_blank(),
            axis_ticks_major_y=element_blank(),
            panel_grid=element_blank(),
            figure_size=figure_size,
        )
    )
    if title:
        p = p + labs(title=title)
    return p


def _tip_labels_layer(
    p: Any,
    ps: Any,
    label_tips: str,
    tip_order: list[str],
    tip_x: dict[str, float],
    tip_y: dict[str, float],
    x_offset: float,
    text_size: float,
    color: str | None,
) -> Any:
    """Add tip-label geom_text for treeonly mode."""
    if ps.tax_table is None:
        raise pyloseqValidationError(
            "label_tips was provided but Phyloseq has no tax_table."
        )
    tax_df = ps.tax_table.to_frame()
    if label_tips not in tax_df.columns:
        raise pyloseqValidationError(
            f"label_tips column '{label_tips}' not found in tax_table."
        )
    tip_lab_df = pd.DataFrame(
        {
            "OTU": tip_order,
            "x_lab": [tip_x[t] + x_offset for t in tip_order],
            "_y": [tip_y[t] for t in tip_order],
            "_label": [
                tax_df.loc[t, label_tips] if t in tax_df.index else ""
                for t in tip_order
            ],
        }
    )
    lab_map: dict[str, str] = {"x": "x_lab", "y": "_y", "label": "_label"}
    if color and color in tip_lab_df.columns:
        lab_map["color"] = color
    return p + geom_text(
        data=tip_lab_df, mapping=aes(**lab_map), ha="left", size=text_size
    )
