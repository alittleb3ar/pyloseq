# Ordination

---

## ordinate

```python
from pyloseq import ordinate

ord_result = ordinate(ps, method="PCoA", distance="bray")
```

Returns an `skbio.stats.ordination.OrdinationResults`. Access sample coordinates via `.samples` (a DataFrame with one row per sample), feature scores via `.features` (not available for all methods), and variance explained per axis via `.proportion_explained`.

**Available methods:**

| Method | Type | Distance required | Formula required |
|---|---|---|---|
| `"PCoA"` / `"MDS"` | Unconstrained | Yes | No |
| `"NMDS"` | Unconstrained | Yes | No |
| `"CA"` | Unconstrained | No | No |
| `"CCA"` | Constrained | No | Yes |
| `"RDA"` | Constrained | No | Yes |
| `"CAP"` | Constrained | Yes | Yes |
| `"DPCoA"` | Unconstrained | Computed from tree | No |
| `"DCA"` | — | — | — |

`"DCA"` is listed for completeness; calling it raises `NotImplementedError`. Use PCoA or CCA instead.

### Unconstrained methods

**PCoA / MDS**

Principal Coordinates Analysis on a distance matrix. The most common choice:

```python
ord_result = ordinate(ps, method="PCoA", distance="bray")
ord_result = ordinate(ps, method="MDS", distance="bray")  # alias
```

**NMDS**

Non-metric Multidimensional Scaling. Uses scikit-bio's `nmds` if available, falls back to scikit-learn's non-metric MDS if not. If neither is installed, falls back to classical metric MDS and emits a warning:

```python
ord_result = ordinate(ps, method="NMDS", distance="jaccard")
```

For NMDS, `proportion_explained` is `NaN`; stress is attached as `ord_result.stress`.

**CA**

Correspondence Analysis on the raw count table (no distance matrix). Produces both sample and feature scores:

```python
ord_result = ordinate(ps, method="CA")
print(ord_result.features)   # taxa coordinates
```

**DPCoA**

Double Principal Coordinates Analysis. Integrates phylogenetic distances; requires `phy_tree`:

```python
ord_result = ordinate(ps, method="DPCoA")
```

### Constrained methods

Constrained ordinations require a `formula` string referencing columns in `sample_data`. Categorical columns are dummy-encoded automatically:

```python
# CCA: one constraining variable
ord_result = ordinate(ps, method="CCA", formula="~SampleType")

# RDA: multiple variables
ord_result = ordinate(ps, method="RDA", formula="~SampleType + Depth")

# CAP: distance-based, constrained
ord_result = ordinate(ps, method="CAP", distance="bray", formula="~SampleType")
```

Terms in the formula must be column names in `sample_data`. Unknown terms raise `pyloseqValidationError`.

### Pre-computed distance matrices

Pass a `skbio.DistanceMatrix` directly to avoid recomputing:

```python
from pyloseq import distance, ordinate

dm = distance(ps, "unifrac")   # expensive; compute once

pcoa_result = ordinate(ps, method="PCoA", distance=dm)
cap_result  = ordinate(ps, method="CAP", distance=dm, formula="~SampleType")
```

### Accessing results

```python
ord_result = ordinate(ps, method="PCoA", distance="bray")

# Sample coordinates (first two axes)
coords = ord_result.samples[["PC1", "PC2"]]

# Variance explained
print(ord_result.proportion_explained[:3])
# PC1    0.312
# PC2    0.187
# PC3    0.089

# Feature scores (available for CA, CCA, RDA)
taxa_coords = ord_result.features
```

::: pyloseq.ordinate
