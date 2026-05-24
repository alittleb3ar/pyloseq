# pyloseq

**pyloseq** is a native Python port of R/Bioconductor [phyloseq](https://joey711.github.io/phyloseq/), built on the PyData stack (pandas, NumPy, scikit-bio, plotnine). It provides first-class QIIME 2 `.qza` support and is designed for researchers migrating 16S/ITS microbiome workflows from R to Python.

Every public function includes an `R reference:` block in its docstring showing the equivalent phyloseq R signature, so your existing R knowledge transfers directly.

## Installation

```bash
pip install pyloseq
```

Requires Python 3.10+.

## Quick start

```python
import pyloseq

# Load from QIIME 2 artifacts
ps = pyloseq.read_qza(
    features="table.qza",
    taxonomy="taxonomy.qza",
    tree="rooted-tree.qza",
    metadata="sample-metadata.tsv",
)

# Or from BIOM v2
ps = pyloseq.read_biom("feature-table.biom")

# Explore
print(ps.ntaxa, ps.nsamples)

# Filter low-abundance taxa, rarefy, and ordinate
threshold = 0.2 * ps.nsamples
ps_filt = pyloseq.filter_taxa(ps, lambda x: (x > 5).sum() > threshold)
ps_rare = pyloseq.rarefy_even_depth(ps_filt, rng_seed=42)

ord_result = pyloseq.ordinate(ps_rare, method="PCoA", distance="bray")
pyloseq.plot_ordination(ps_rare, ord_result, color="SampleType")
```

## What's included

| Category | Functionality |
|----------|--------------|
| **Containers** | `Phyloseq`, `OtuTable`, `SampleData`, `TaxTable`, `PhyTree`, `RefSeq` |
| **I/O** | BIOM v2, QIIME 2 `.qza`, QIIME 1, mothur, CSV |
| **Manipulation** | subset, prune, filter, rarefy, agglomerate, merge, melt |
| **Diversity** | Alpha (9 measures: Shannon, Chao1, ACE, Fisher, …), Beta (30+ distances), UniFrac |
| **Ordination** | PCoA, NMDS, MDS, CCA, RDA, CAP |
| **Plotting** | bar, richness, ordination, heatmap, network (plotnine/ggplot2 API) |

## Documentation

- [Full documentation](https://mikedonovan.github.io/pyloseq)
- [API reference](https://mikedonovan.github.io/pyloseq/api/pyloseq/)
- [Tutorial notebooks](https://mikedonovan.github.io/pyloseq/notebooks/)
- [R → Python migration guide](https://mikedonovan.github.io/pyloseq/api/pyloseq/)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
