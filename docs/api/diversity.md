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
| `PD` | Faith's phylogenetic diversity (Faith 1992); requires `phy_tree` |

Pass `measures=None` (the default) to compute all available measures. When `ps.phy_tree` is `None`, `PD` is excluded from the defaults. Unrecognized measure names raise `pyloseqValidationError`.

**Pooled richness:**

Pass `split=False` to pool all samples into a single community before computing:

```python
pooled = estimate_richness(ps, split=False)
# Returns a single row labelled "pooled"
```

**Faith's Phylogenetic Diversity:**

```python
# PD requires a phylogenetic tree on the Phyloseq object
pd_df = estimate_richness(ps_with_tree, measures=["PD"])

# Mix PD with standard measures in one call
alpha = estimate_richness(ps_with_tree, measures=["Observed", "Shannon", "PD"])
```

`PD` is the sum of branch lengths in the minimum spanning tree connecting all observed taxa plus the root. The tree is midpoint-rooted internally if unrooted, matching the convention used by R's `phangorn::midpoint`.

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

### gunifrac

Computes the Generalized UniFrac family of distance matrices (Chen et al. 2012
*Bioinformatics* 28:2106–2113), matching the R `GUniFrac` package API:

```python
from pyloseq import gunifrac

results = gunifrac(ps)                   # default alpha=(0, 0.5, 1)
dm_half = results["d_0.5"]              # GUniFrac at alpha=0.5
dm_uw   = results["d_UW"]               # unweighted UniFrac
dm_vaw  = results["d_VAW"]              # variance-adjusted weighted UniFrac
```

**Return value** — a `dict` of `skbio.stats.distance.DistanceMatrix` objects:

| Key | Description |
|---|---|
| `"d_{a}"` | GUniFrac at exponent *a* for each value in `alpha` (e.g. `"d_0.5"`) |
| `"d_UW"` | Unweighted UniFrac (Chen 2012 definition) |
| `"d_VAW"` | Variance-adjusted weighted UniFrac (Hamady et al. 2010) |

Alpha = 0 up-weights rare lineages; alpha = 1 is equivalent to normalized weighted
UniFrac. The default `alpha=(0, 0.5, 1)` covers the full range.

**Piping into `make_network`:**

```python
from pyloseq import make_network, plot_network

g = make_network(ps, distance=results["d_0.5"], max_dist=0.5)
p = plot_network(g, ps, color="SampleType")
```

!!! note
    `d_UW` from `gunifrac` matches R's `GUniFrac` package definition, which counts any
    branch whose cumulative proportion differs between the two samples (including
    branches shared by both but at different abundances). This differs slightly from
    `unifrac()`, which uses the Lozupone & Knight (2005) definition counting only
    branches exclusive to one sample.

::: pyloseq.gunifrac

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
