# Plotting

All plot functions return `plotnine.ggplot` objects, which can be composed with `+` before rendering. Call `.draw()` to display or `.save("output.pdf")` to write to disk.

The `make_network` function is an exception: it returns a `networkx.Graph`.

```python
from pyloseq import (
    plot_bar, plot_richness, plot_ordination,
    plot_heatmap, plot_tree, make_network, plot_network,
)
```

---

## plot_bar

Stacked bar chart of OTU abundances. Internally calls `psmelt` to convert to long format before plotting.

```python
p = plot_bar(ps, fill="Phylum", facet_grid="~ SampleType")
p.draw()
```

Stack order is deterministic: sorted first by the `x` column, then by the `fill` column, matching R phyloseq's rendering.

```python
# Group by sample type on x-axis, fill by phylum
p = plot_bar(ps, x="SampleType", fill="Phylum")

# Facet by sample type
p = plot_bar(ps, fill="Genus", facet_grid="SampleType ~")

# Compose with a plotnine theme
from plotnine import theme_bw
p = plot_bar(ps, fill="Phylum") + theme_bw()
```

::: pyloseq.plot_bar

---

## plot_richness

Alpha diversity plot faceted by measure. By default each panel shows a box-and-whisker summary; set `boxplot=False` for a points-only view (useful when `x` groups only a few samples and boxes add noise):

```python
p = plot_richness(ps, x="SampleType", color="SampleType",
                  measures=["Shannon", "Simpson"])
p.draw()

# Points only — no boxes
p = plot_richness(ps, x="SampleType", color="SampleType", boxplot=False)
```

Standard-error whiskers are drawn automatically for measures that have a corresponding `se.*` column in the `estimate_richness` output (currently only `se.chao1`).

Setting `x="samples"` (the default) puts individual sample names on the x-axis. Set `x` to any column in `sample_data` to group or aggregate:

```python
# Individual samples, coloured by environment type
p = plot_richness(ps, x="samples", color="SampleType")

# Boxplots by environment type
p = plot_richness(ps, x="SampleType", color="SampleType")
```

::: pyloseq.plot_richness

---

## plot_ordination

Scatter plot of ordination results. The `kind` parameter controls what is plotted:

| `kind` | What is plotted |
|---|---|
| `"samples"` | Sample coordinates (default) |
| `"taxa"` | Feature/taxa scores (requires ordination with feature scores, e.g. CA, RDA) |
| `"biplot"` | Samples and taxa together, taxa scores rescaled to sample axis range |
| `"split"` | Samples and taxa side-by-side in facets |
| `"scree"` | Proportion of variance explained per axis |

```python
from pyloseq import ordinate, plot_ordination

ord_result = ordinate(ps, method="PCoA", distance="bray")

# Basic sample scatter
p = plot_ordination(ps, ord_result, color="SampleType")
p.draw()

# With convex hulls per colour group
p = plot_ordination(ps, ord_result, color="SampleType", show_hull=True)

# Scree plot
p = plot_ordination(ps, ord_result, kind="scree")

# Return the DataFrame instead of a plot (for custom plotting)
df = plot_ordination(ps, ord_result, just_df=True)
```

!!! note
    `kind="taxa"` and `kind="biplot"` require an ordination that produces feature scores, such as CA, CCA, or RDA. PCoA and NMDS do not produce feature scores; using those with `kind="taxa"` raises `pyloseqValidationError`.

::: pyloseq.plot_ordination

---

## plot_heatmap

Heatmap of OTU abundances across samples. By default, rows and columns are reordered by an ordination to group similar taxa and samples together:

```python
p = plot_heatmap(ps, method="PCoA", distance="bray", trans="log4")
p.draw()
```

Pass `method=None` to skip ordination entirely and preserve the original sample and taxa order from the `Phyloseq` object:

```python
p = plot_heatmap(ps, method=None)
```

The `trans` parameter applies a transformation before plotting. Zero values become `NaN` and render using `na_value` rather than the low-end gradient colour:

| `trans` | Transformation |
|---|---|
| `None` | Raw values |
| `"log4"` | log₄(x), zeros → `na_value` |

Use `label` to replace x-axis sample-name tick labels with a `sample_data` variable, and `taxa_label` to replace y-axis taxa-name tick labels with a taxonomic rank. Ordering (from ordination or original order) is unaffected — only the tick text changes. These mirror R phyloseq's `sample.label` and `taxa.label`:

```python
# Label x-axis ticks by treatment group, y-axis ticks by phylum
p = plot_heatmap(ps, method="PCoA", label="TreatmentGroup", taxa_label="Phylum")
```

::: pyloseq.plot_heatmap

---

## plot_tree

Phylogenetic tree visualization. Requires `phy_tree`:

```python
p = plot_tree(ps, color="SampleType", label_tips="Phylum")
p.draw()
```

**`method` parameter:**

- `"treeonly"` — draw the tree structure only
- `"sampledodge"` — dodge sample points along the tips by metadata

```python
# Tree with phylum labels at tips
p = plot_tree(ps, method="treeonly", label_tips="Phylum", ladderize=True)

# Sample-dodge mode: show samples by environment type at tips
p = plot_tree(ps, method="sampledodge", color="SampleType", size="Abundance")
```

::: pyloseq.plot_tree

---

## make_network / plot_network

`make_network` builds a `networkx.Graph` where nodes are samples and edges connect samples whose distance is below `max_dist`:

```python
from pyloseq import make_network, plot_network

g = make_network(ps, max_dist=0.4, distance="bray")
p = plot_network(g, ps, color="SampleType", label="SampleID")
p.draw()
```

The `distance` parameter accepts either a metric name string (see [`distance`](diversity.md#distance)) **or** a precomputed `skbio.stats.distance.DistanceMatrix` — mirroring R's `plot_net(distance=as.dist(...))` pattern. This makes it straightforward to use phylogenetic distances from `gunifrac`:

```python
from pyloseq import gunifrac, make_network, plot_network

results = gunifrac(ps)
g = make_network(ps, distance=results["d_0.5"], max_dist=0.5)
p = plot_network(g, ps, color="SampleType")
p.draw()
```

**Edge width** is scaled inversely by distance: edges between more-similar samples appear thicker, matching R's `plot_net`. If the `shape` aesthetic maps to more than 6 unique values, the shape legend is suppressed automatically (shapes are still distinct in the plot) and a warning is issued — plotnine's default shape palette has only 6 entries.

Node attributes from `sample_data` are attached to each node automatically, making the graph available for further networkx analysis.

::: pyloseq.make_network

::: pyloseq.plot_network
