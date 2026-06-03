# Manipulation

All manipulation functions return new `Phyloseq` objects; inputs are never modified. Every operation propagates all components that are not directly affected — so filtering taxa keeps the sample data unchanged, filtering samples keeps the tax table unchanged, and so on.

---

## Subsetting

### prune_taxa / prune_samples

Take an explicit list of names. Names not present in the object are silently ignored; the output preserves the order of the input list:

```python
from pyloseq import prune_taxa, prune_samples

ps2 = prune_taxa(["OTU1", "OTU3", "OTU9"], ps)
ps2 = prune_samples(["S1", "S4", "S7"], ps)
```

Use these when you already have the names. Use `subset_*` when you need to filter by a condition.

::: pyloseq.prune_taxa

::: pyloseq.prune_samples

### subset_taxa / subset_samples

Filter by a predicate applied to the tax table or sample data. The predicate can be a callable or a pandas query string:

```python
from pyloseq import subset_taxa, subset_samples

# Callable form — receives one row as a Series
ps2 = subset_taxa(ps, lambda t: t["Phylum"] == "Firmicutes")
ps2 = subset_samples(ps, lambda s: s["SampleType"] == "Soil")

# Query string form
ps2 = subset_taxa(ps, 'Phylum == "Bacteroidetes"')
ps2 = subset_samples(ps, 'SampleType == "Ocean" and Depth > 100')
```

`subset_taxa` requires `tax_table`; `subset_samples` requires `sample_data`.

For the string form, column names with spaces must be backtick-quoted: `` 'Consensus\ Lineage == "Bacteria"' `` → `` '`Consensus Lineage` == "Bacteria"' ``. Use the callable form to avoid this.

::: pyloseq.subset_taxa

::: pyloseq.subset_samples

---

## Filtering

### filter_taxa

Applies a predicate to each taxon's abundance vector (one value per sample). Returns a pruned `Phyloseq` with only the taxa where the predicate returns `True`:

```python
from pyloseq import filter_taxa

# Keep taxa with mean abundance > 0.001
ps2 = filter_taxa(ps, lambda x: x.mean() > 0.001)
```

This is equivalent to R's `filter_taxa(physeq, flist, prune=TRUE)`. To get the boolean mask without pruning, use `taxa_filter_mask`.

::: pyloseq.filter_taxa

### kOverA

Factory for a common filter: keep taxa present in at least *k* samples with abundance greater than *A*:

```python
from pyloseq import kOverA, filter_taxa

# Keep taxa with raw count > 10 in at least 3 samples
ps2 = filter_taxa(ps, kOverA(3, 10))
```

::: pyloseq.kOverA

### taxa_filter_mask

Returns the boolean `pd.Series` from a predicate without pruning. Useful for inspecting which taxa would be removed before committing:

```python
from pyloseq import taxa_filter_mask, kOverA

mask = taxa_filter_mask(ps, kOverA(3, 10))
print(mask.sum(), "taxa would be kept")
```

::: pyloseq.taxa_filter_mask

---

## Transformation

### transform_sample_counts

Applies a function column-by-column across the OTU table. Each call receives a `pd.Series` of abundances for one sample (indexed by taxa name) and must return a Series of equal length:

```python
from pyloseq import transform_sample_counts
import numpy as np

# Relative abundance
ps_rel = transform_sample_counts(ps, lambda x: x / x.sum())

# Log-transform (adding a pseudocount)
ps_log = transform_sample_counts(ps, lambda x: np.log1p(x))
```

!!! warning
    Samples where the total count is zero will produce `NaN` or `inf` after division. Filter out zero-count samples before normalizing, or handle them explicitly in the function.

::: pyloseq.transform_sample_counts

### rarefy_even_depth

Random subsampling (rarefaction) to a uniform sequencing depth:

```python
from pyloseq import rarefy_even_depth

ps_rare = rarefy_even_depth(ps, sample_size=10000, replace=False, rng_seed=42)
```

Samples with fewer than `sample_size` counts are dropped. The default `sample_size` is the minimum sample sum in the dataset. Set `replace=True` for sampling with replacement (not recommended for most analyses).

::: pyloseq.rarefy_even_depth

---

## Merging

### merge_phyloseq

Combines multiple `Phyloseq` objects by taking the union of taxa and samples, summing counts where both are present:

```python
from pyloseq import merge_phyloseq

merged = merge_phyloseq(ps1, ps2, ps3)
```

::: pyloseq.merge_phyloseq

### merge_samples

Collapses samples that share the same value of a metadata variable. Abundance counts are summed across samples within each group; numeric metadata columns are averaged; non-numeric columns that are constant within a group are retained, others become `NaN`:

```python
from pyloseq import merge_samples

# One row per SampleType
ps_grouped = merge_samples(ps, "SampleType")
```

::: pyloseq.merge_samples

### merge_taxa

Collapses a list of taxa into a single representative, summing their counts. The `archetype` parameter names which taxon inherits the taxonomy annotation and reference sequence of the merged group:

```python
from pyloseq import merge_taxa

ps2 = merge_taxa(ps, ["OTU1", "OTU2", "OTU7"], archetype="OTU1")
```

::: pyloseq.merge_taxa

---

## Aggregation

### tax_glom

Collapses taxa that share the same annotation at a given taxonomic rank. Abundance counts are summed within each unique value. This is the primary way to work at phylum or genus level:

```python
from pyloseq import tax_glom

ps_phylum = tax_glom(ps, "Phylum")
ps_genus  = tax_glom(ps, "Genus")
```

Taxa with `NaN` at the target rank are collected into a single `"Unknown"` bin by default. Pass `na_rm=True` to drop them instead.

::: pyloseq.tax_glom

### tip_glom

Collapses taxa that are within a given patristic distance of each other on the phylogenetic tree. Requires `phy_tree`:

```python
from pyloseq import tip_glom

ps2 = tip_glom(ps, h=0.05)
```

::: pyloseq.tip_glom

---

## Reshaping

### psmelt

Converts from wide format (taxa × samples matrix) to long format (one row per taxon-sample combination). The output DataFrame includes columns for `OTU`, `Sample`, `Abundance`, all sample metadata variables, and all taxonomic rank columns:

```python
from pyloseq import psmelt

long_df = psmelt(ps)
# Columns: OTU, Sample, Abundance, Kingdom, Phylum, ..., SampleType, ...
```

This is the same as calling `ps.melt()`. Use the resulting DataFrame for custom ggplot layers or non-phyloseq statistical tests.

::: pyloseq.psmelt
