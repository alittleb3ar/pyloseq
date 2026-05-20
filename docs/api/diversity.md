# Diversity

---

## Alpha diversity

### estimate_richness

Estimates within-sample diversity. Returns a DataFrame indexed by sample name:

```python
from pyloseq import estimate_richness

alpha = estimate_richness(ps, measures=["Observed", "Chao1", "Shannon"])
```

**Available measures:**

| Measure | Description |
|---|---|
| `Observed` | Count of taxa with non-zero abundance |
| `Chao1` | Chao1 richness estimator (Chao 1984) |
| `se.chao1` | Standard error of Chao1 |
| `ACE` | Abundance Coverage Estimator (Chao & Lee 1992); matches vegan |
| `se.ACE` | SE of ACE — not implemented, always `NaN` |
| `Shannon` | Shannon entropy (*H* = −Σ p log p) |
| `Simpson` | Simpson's diversity (1 − Σ p²) |
| `InvSimpson` | Inverse Simpson (1 / Σ p²) |
| `Fisher` | Fisher's log-series alpha |

Pass `measures=None` (the default) to compute all nine. Unrecognized measure names raise `pyloseqValidationError`.

**Pooled richness:**

Pass `split=False` to pool all samples into a single community before computing:

```python
pooled = estimate_richness(ps, split=False)
# Returns a single row labelled "pooled"
```

!!! note
    Chao1, ACE, Observed, and Fisher are count-based; pass raw integer counts, not relative abundances. Applying `estimate_richness` after `transform_sample_counts` will produce incorrect estimates for those measures.

::: pyloseq.estimate_richness

---

## Beta diversity

### distance

Computes a pairwise distance matrix between samples (or taxa). Returns an `skbio.stats.distance.DistanceMatrix`:

```python
from pyloseq import distance

dm = distance(ps, "bray")
dm = distance(ps, "unifrac")
dm = distance(ps, "jaccard", kind="samples")
```

**Available methods:**

| Method | Backend | Notes |
|---|---|---|
| `"bray"` | scipy | Bray-Curtis dissimilarity |
| `"jaccard"` | scipy | Binary Jaccard (presence/absence) |
| `"euclidean"` | scipy | |
| `"manhattan"` | scipy | City-block / L1 |
| `"canberra"` | scipy | |
| `"minkowski"` | scipy | Pass `p=` to control exponent |
| `"cosine"` | scipy | |
| `"correlation"` | scipy | Pearson correlation distance |
| `"maximum"` | scipy | Chebyshev / L∞ |
| `"binary"` | scipy | Synonym for `"jaccard"` |
| `"sorensen"` | scipy | Sørensen-Dice (presence/absence) |
| `"unifrac"` | scikit-bio | Unweighted UniFrac; requires `phy_tree` |
| `"wunifrac"` | scikit-bio | Weighted UniFrac; requires `phy_tree` |
| `"jsd"` | scipy | Jensen-Shannon divergence (√JSD, base 2) |
| `"dpcoa"` | custom | Double PCoA patristic distance; requires `phy_tree` |

**`kind` parameter:**

`kind="samples"` (default) computes an *n_samples × n_samples* matrix. `kind="taxa"` transposes before computing, yielding an *n_taxa × n_taxa* matrix. Most phylogenetic methods (`unifrac`, `wunifrac`, `dpcoa`) only support `kind="samples"`.

**Passing kwargs to scipy:**

```python
# Minkowski with p=1 (= Manhattan)
dm = distance(ps, "minkowski", p=1)
```

::: pyloseq.distance

### unifrac

Direct interface to UniFrac, bypassing the `distance` dispatcher:

```python
from pyloseq import unifrac

dm_uw = unifrac(ps, weighted=False)
dm_w  = unifrac(ps, weighted=True, normalized=True)
```

`normalized=True` divides by total branch length; this only affects weighted UniFrac. The `n_jobs` parameter controls parallelism in the scikit-bio implementation.

::: pyloseq.unifrac

### distance_method_list

Returns all supported methods grouped by backend:

```python
from pyloseq import distance_method_list

methods = distance_method_list()
# {
#   "phylogenetic":    ["dpcoa", "unifrac", "wunifrac"],
#   "information":     ["jsd"],
#   "vegan-equivalent": ["bray", "canberra", ...]
# }
```

::: pyloseq.distance_method_list
