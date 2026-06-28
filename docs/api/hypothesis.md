# Hypothesis Testing

---

## multi_tax_test

Per-taxon differential abundance test between two groups. Applies a statistical test to each taxon independently, then corrects for multiple comparisons:

```python
from pyloseq import multi_tax_test

results = multi_tax_test(ps, grouping_var="SampleType", test="t", method="BH")
print(results.head(10))
```

The function requires `sample_data` with a column that has exactly two distinct non-NaN values. Samples with NaN in the grouping column are dropped silently.

### Test statistics

| `test` | Method |
|---|---|
| `"t"` | Welch's t-test (`equal_var=False`) |
| `"wilcoxon"` | Wilcoxon rank-sum test |

Welch's t-test is appropriate when group variances may differ and both groups have at least a few samples. The Wilcoxon rank-sum test is a non-parametric alternative; use it when count distributions are highly skewed or sample sizes are very small.

### Multiple-testing correction

| `method` | Type | Description |
|---|---|---|
| `"BH"` | FDR | Benjamini-Hochberg (default). Controls false discovery rate. |
| `"BY"` | FDR | Benjamini-Yekutieli. More conservative than BH; valid under arbitrary correlation. |
| `"holm"` | FWER | Holm step-down. Controls family-wise error rate without assuming independence. |
| `"bonferroni"` | FWER | Bonferroni. Most conservative; appropriate when any false positive is unacceptable. |
| `"westfall_young"` | FWER | Permutation-based step-down (Westfall & Young 1993). Equivalent to R's `multtest::mt.minP`. Respects the correlation structure of the test statistics. |

### Return value

A DataFrame with one row per taxon, sorted by ascending `adjp`:

| Column | Description |
|---|---|
| `statistic` | Per-taxon test statistic |
| `rawp` | Uncorrected p-value |
| `adjp` | Corrected p-value |
| `mean_<group1>` | Mean abundance in group 1 |
| `mean_<group2>` | Mean abundance in group 2 |

Group names in the mean columns come from the sorted unique values of `grouping_var`.

### Examples

```python
# Default: Welch t-test, BH correction
results = multi_tax_test(ps, "SampleType")
significant = results[results["adjp"] < 0.05]

# Wilcoxon with Holm FWER control
results = multi_tax_test(ps, "SampleType", test="wilcoxon", method="holm")

# Permutation FWER — use more permutations for stable estimates
results = multi_tax_test(
    ps, "SampleType",
    method="westfall_young",
    n_permutations=5000,
    rng_seed=0,
)
```

!!! note
    `westfall_young` runs `n_permutations` separate tests per permutation and scales as O(n_taxa × n_permutations). For datasets with tens of thousands of taxa, use a smaller `n_permutations` (e.g. 500–1000) for exploration and increase it only for final analysis.

::: pyloseq.multi_tax_test

---

## permanova

PERMANOVA (Permutational Multivariate Analysis of Variance) tests whether the centroids of two or more groups differ in multivariate space. Thin wrapper around `skbio.stats.distance.permanova` that extracts group labels from `sample_data` automatically:

```python
from pyloseq import distance, permanova

dm = distance(ps, "bray")
result = permanova(dm, ps, grouping_var="SampleType", permutations=999)
print(result["p-value"])
print(result["test statistic"])  # pseudo-F
```

The distance matrix and the Phyloseq object do not need to have identical sample sets. Only samples present in `distance_matrix.ids` are used; the rest of `ps` is ignored. This means you can compute a distance matrix on a filtered subset and still pass the original `ps`:

```python
ps_sub = subset_samples(ps, ps.sample_data.to_frame()["Env"] == "Soil")
dm_sub = distance(ps_sub, "bray")
result = permanova(dm_sub, ps_sub, "Treatment")
```

The return value is a `pd.Series` from scikit-bio with keys `method name`, `test statistic name`, `sample size`, `number of groups`, `test statistic`, `p-value`, and `number of permutations`.

R reference: `vegan::adonis2(dist ~ group, data = sample_data(physeq))`

::: pyloseq.permanova

---

## betadisper

PERMDISP test for homogeneity of multivariate dispersions. Tests whether groups have similar spread around their centroids — a complement to PERMANOVA that checks the equal-dispersion assumption. Thin wrapper around `skbio.stats.distance.permdisp`:

```python
from pyloseq import betadisper, distance

dm = distance(ps, "bray")
result = betadisper(dm, ps, grouping_var="SampleType", permutations=999)
print(result["p-value"])
```

Use `betadisper` together with `permanova` to distinguish centroid differences (PERMANOVA) from dispersion differences (betadisper):

```python
dm = distance(ps, "bray")
perm = permanova(dm, ps, "Treatment")
disp = betadisper(dm, ps, "Treatment")

# Significant PERMANOVA + non-significant betadisper → true centroid shift
# Significant betadisper → group variances differ (confounds PERMANOVA)
```

R reference: `vegan::betadisper()` + `vegan::permutest()`

::: pyloseq.betadisper
