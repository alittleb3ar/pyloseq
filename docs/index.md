# pyloseq — Python Microbiome Analysis (phyloseq port)

**pyloseq** is a Python port of R's [phyloseq](https://joey711.github.io/phyloseq/) Bioconductor package for microbiome data analysis. Built on the PyData stack (NumPy, pandas, scikit-bio), it provides the same workflow — BIOM/QIIME2 import, alpha/beta diversity, ordination, and publication-quality plots — without leaving Python.

Microbiome data is represented as a single `Phyloseq` object bundling an OTU/feature table with optional sample metadata, taxonomic annotations, a phylogenetic tree, and reference sequences. All analysis functions — diversity, ordination, hypothesis testing, plotting — operate on that object directly. Every public function includes an `R reference:` block in its docstring showing the equivalent phyloseq signature.

## Installation

```bash
pip install pyloseq
```

Requires Python 3.10+.

## Dependencies

| Package | Minimum version | Role |
|---|---|---|
| numpy | 1.24 | Numerical core |
| pandas | 2.0 | DataFrames |
| scipy | 1.11 | Distances, statistics |
| scikit-bio | 0.7 | Ordination, UniFrac, tree I/O |
| plotnine | 0.13 | ggplot2-style plotting |
| biom-format | 2.1 | BIOM file I/O |
| h5py | 3.9 | BIOM v2 (HDF5) support |
| pyarrow | 12 | Parquet I/O for golden files |
| pyyaml | 6 | QIIME 2 metadata parsing |

## Hello world

```python
import pyloseq as ps
from pyloseq import (
    Phyloseq, OtuTable, SampleData, TaxTable,
    filter_taxa, kOverA, transform_sample_counts,
    estimate_richness, distance, ordinate,
    plot_richness, plot_ordination,
)
from pyloseq.datasets import load_global_patterns_reference

ref = load_global_patterns_reference()

gp = Phyloseq(
    otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
    sam=SampleData(ref["sample_data"]),
    tax=TaxTable(ref["tax_table"]),
)

# Keep taxa present in at least 5 samples
gp = filter_taxa(gp, kOverA(5, 0))

# Relative abundance
gp_rel = transform_sample_counts(gp, lambda x: x / x.sum())

# Alpha diversity
alpha = estimate_richness(gp, measures=["Shannon", "Simpson"])

# Ordination
dm = distance(gp, "bray")
ord_result = ordinate(gp, method="PCoA", distance=dm)

plot_richness(gp, x="SampleType", color="SampleType").draw()
plot_ordination(gp, ord_result, color="SampleType").draw()
```
