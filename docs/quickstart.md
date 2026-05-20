# Quickstart

This walkthrough covers a complete microbiome analysis pipeline using the built-in `GlobalPatterns` dataset — 26 samples across several environment types, 19,216 taxa, with a phylogenetic tree. All the code shown here runs without any external files.

## Loading data

The `load_global_patterns_reference()` function returns a dict of DataFrames and a Newick string. Wrap each component in its container class and pass them to `Phyloseq`:

```python
from pyloseq import Phyloseq, OtuTable, SampleData, TaxTable, PhyTree
from pyloseq.datasets import load_global_patterns_reference

ref = load_global_patterns_reference()

gp = Phyloseq(
    otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
    sam=SampleData(ref["sample_data"]),
    tax=TaxTable(ref["tax_table"]),
    tree=PhyTree.from_newick(ref["phy_tree_newick"]),
)
```

Inspect the object:

```python
print(gp.ntaxa, gp.nsamples)       # 19216 26
print(gp.sample_variables)         # ['SampleType', 'Description', ...]
print(gp.rank_names)               # ['Kingdom', 'Phylum', 'Class', ...]
```

## Filtering low-abundance taxa

Most 16S datasets have a long tail of rare taxa. `filter_taxa` combined with `kOverA` keeps only taxa present in at least *k* samples:

```python
from pyloseq import filter_taxa, kOverA

# Keep taxa with count > 0 in at least 5 samples
gp = filter_taxa(gp, kOverA(5, 0))
print(gp.ntaxa)   # reduced to ~4000
```

`kOverA(k, A)` returns a predicate; you can substitute any function that accepts a `pd.Series` of per-sample abundances and returns `bool`.

## Relative-abundance normalization

Absolute counts are not comparable across samples with different sequencing depth. Transform to relative abundances before computing beta diversity:

```python
from pyloseq import transform_sample_counts

gp_rel = transform_sample_counts(gp, lambda x: x / x.sum())
```

`transform_sample_counts` applies the function column-by-column (one column = one sample) and returns a new `Phyloseq` with the same components but a replaced OTU table.

## Alpha diversity

`estimate_richness` returns a DataFrame indexed by sample name. Pass a `measures` list to select specific metrics, or omit it to get all nine:

```python
from pyloseq import estimate_richness

alpha = estimate_richness(gp, measures=["Observed", "Chao1", "Shannon", "Simpson"])
print(alpha.head())
```

```
                Observed   Chao1    Shannon   Simpson
CL3             340.0    345.2    5.21      0.987
CC1             324.0    332.7    4.98      0.983
SV1             245.0    249.1    4.56      0.979
...
```

!!! note
    Pass integer count data to `estimate_richness`. Chao1 and ACE are count-based estimators; fractional counts from rarefaction or relative-abundance tables give nonsensical results.

## Beta diversity

`distance` computes a pairwise `skbio.DistanceMatrix`. The most common choice for 16S is Bray-Curtis:

```python
from pyloseq import distance

dm = distance(gp, "bray")
```

For phylogenetically weighted distances, pass `"unifrac"` or `"wunifrac"` — these require a `phy_tree`:

```python
dm_uf = distance(gp, "unifrac")
dm_wuf = distance(gp, "wunifrac")
```

`distance_method_list()` returns all available methods grouped by backend.

## Ordination

Pass the distance matrix (or a method string) to `ordinate`:

```python
from pyloseq import ordinate

ord_result = ordinate(gp, method="PCoA", distance=dm)
```

The return value is an `skbio.OrdinationResults` object. Access sample coordinates and variance explained:

```python
print(ord_result.samples.head())          # DataFrame, one row per sample
print(ord_result.proportion_explained)    # variance per axis
```

## Plotting

All plot functions return `plotnine.ggplot` objects, which can be composed with `+` before drawing:

```python
from pyloseq import plot_richness, plot_ordination, plot_bar

# Alpha diversity panel, grouped by environment type
p_alpha = plot_richness(gp, x="SampleType", color="SampleType",
                        measures=["Shannon", "Simpson"])
p_alpha.draw()

# PCoA scatter
p_ord = plot_ordination(gp, ord_result, color="SampleType", show_hull=True)
p_ord.draw()

# Stacked bar at Phylum level (relative abundance)
from pyloseq import tax_glom
gp_phylum = tax_glom(gp_rel, "Phylum")
p_bar = plot_bar(gp_phylum, fill="Phylum", facet_grid="~ SampleType")
p_bar.draw()
```

## Complete pipeline

```python
from pyloseq import (
    Phyloseq, OtuTable, SampleData, TaxTable, PhyTree,
    filter_taxa, kOverA, transform_sample_counts, tax_glom,
    estimate_richness, distance, ordinate,
    plot_richness, plot_ordination, plot_bar,
)
from pyloseq.datasets import load_global_patterns_reference

ref = load_global_patterns_reference()

gp = Phyloseq(
    otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
    sam=SampleData(ref["sample_data"]),
    tax=TaxTable(ref["tax_table"]),
    tree=PhyTree.from_newick(ref["phy_tree_newick"]),
)

# Filter and normalise
gp = filter_taxa(gp, kOverA(5, 0))
gp_rel = transform_sample_counts(gp, lambda x: x / x.sum())

# Alpha
alpha = estimate_richness(gp, measures=["Shannon", "Simpson"])

# Beta and ordination
dm = distance(gp, "bray")
ord_result = ordinate(gp, method="PCoA", distance=dm)

# Plots
plot_richness(gp, x="SampleType", color="SampleType").draw()
plot_ordination(gp, ord_result, color="SampleType").draw()
plot_bar(tax_glom(gp_rel, "Phylum"), fill="Phylum").draw()
```
