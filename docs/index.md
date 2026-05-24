# pyloseq

**pyloseq** is a native Python port of R/Bioconductor [phyloseq](https://joey711.github.io/phyloseq/), built on the PyData stack (pandas, NumPy, scikit-bio, plotnine). It provides QIIME 2 `.qza` support out of the box — something R phyloseq lacks — and is designed for users migrating 16S/ITS microbiome workflows from R to Python.

## Quick start

```python
import pyloseq

# Load from BIOM v2
ps = pyloseq.read_biom("feature-table.biom")

# Load from QIIME 2
ps = pyloseq.read_qza(features="table.qza", taxonomy="taxonomy.qza", tree="rooted-tree.qza")

# Explore
print(ps.ntaxa, ps.nsamples)
print(ps.taxa_names[:5])

# Rarefy
ps_rare = pyloseq.rarefy_even_depth(ps, sample_size=10_000, rng_seed=42)

# Ordinate
ord_result = pyloseq.ordinate(ps_rare, method="PCoA", distance="bray")
pyloseq.plot_ordination(ps_rare, ord_result, color="SampleType")
```

## R → Python migration

Every public function includes an `R reference:` block in its docstring showing the equivalent phyloseq R signature. See the [API reference](api/pyloseq.md) for the full migration table.

## Installation

```bash
pip install pyloseq
```
