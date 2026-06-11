# pyloseq

**[Documentation →](https://alittleb3ar.github.io/pyloseq/)**

A Python port of the R/Bioconductor [phyloseq](https://joey711.github.io/phyloseq/) package, built on the PyData stack. pyloseq represents microbiome data as a single object that bundles an OTU/feature table with sample metadata, taxonomic annotations, a phylogenetic tree, and reference sequences. All analysis functions operate on that object directly.

Designed for researchers migrating 16S/ITS workflows from R to Python. Every public function includes an `R reference:` block in its docstring.

## Installation

```bash
pip install pyloseq
```

Requires Python 3.10+.

## Using in containers

pyloseq installs with `pip` on any standard Python base image:

```bash
docker run --rm python:3.12-slim sh -c "pip install pyloseq && python -c 'import pyloseq; print(pyloseq.__version__)'"
```

Tested base images: `python:3.10-slim`, `python:3.11-slim`, `python:3.12-slim`, `python:3.13-slim`, `continuumio/miniconda3`, `jupyter/scipy-notebook`. See [docs/containers.md](docs/containers.md) for a minimal Dockerfile, conda setup, and more.

## Quick start

```python
from pyloseq import (
    Phyloseq, OtuTable, SampleData, TaxTable, PhyTree,
    filter_taxa, kOverA, transform_sample_counts,
    estimate_richness, distance, ordinate,
    plot_richness, plot_ordination,
)
from pyloseq.datasets import load_global_patterns_reference

ref = load_global_patterns_reference()

ps = Phyloseq(
    otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
    sam=SampleData(ref["sample_data"]),
    tax=TaxTable(ref["tax_table"]),
    tree=PhyTree.from_newick(ref["phy_tree_newick"]),
)

# Filter rare taxa, normalize to relative abundance
ps = filter_taxa(ps, kOverA(5, 0))
ps_rel = transform_sample_counts(ps, lambda x: x / x.sum())

# Alpha diversity
alpha = estimate_richness(ps, measures=["Shannon", "Simpson"])

# Bray-Curtis PCoA
dm = distance(ps, "bray")
ord_result = ordinate(ps, method="PCoA", distance=dm)

plot_richness(ps, x="SampleType", color="SampleType").draw()
plot_ordination(ps, ord_result, color="SampleType").draw()
```

## Features

**Data containers**
- `Phyloseq` — top-level object bundling OTU table, sample metadata, taxonomy, tree, and reference sequences
- Automatic pruning to the intersection of taxa/sample names across components on construction
- Sparse OTU table storage (auto-detected below 50% density)

**Input / Output**
- BIOM v1 (JSON) and v2 (HDF5) — read and write
- QIIME 2 `.qza` artifacts — no `qiime2` package required
- QIIME 1 legacy OTU tables and mapping files
- mothur `.shared`, `.list` + `.group`, `.cons.taxonomy`
- Plain CSV/TSV

**Manipulation** (all functions return new objects; inputs are never modified)
- `prune_taxa`, `prune_samples` — subset by explicit name list
- `subset_taxa`, `subset_samples` — filter by callable or pandas query string
- `filter_taxa`, `kOverA` — abundance-based filtering
- `transform_sample_counts` — apply any per-sample function (normalization, log transform, etc.)
- `rarefy_even_depth` — random subsampling to uniform depth
- `tax_glom` — collapse taxa to a given rank
- `tip_glom` — collapse by phylogenetic distance
- `merge_phyloseq`, `merge_samples`, `merge_taxa`
- `psmelt` — wide to long (tidy) format

**Diversity**
- Alpha: Observed, Chao1, ACE, Shannon, Simpson, InvSimpson, Fisher
- Beta: Bray-Curtis, Jaccard, UniFrac, weighted UniFrac, JSD, DPCoA, and all scipy distance metrics
- `distance` dispatcher accepts method strings or pre-computed `skbio.DistanceMatrix`

**Ordination**
- PCoA / MDS, NMDS, CA, CCA, RDA, CAP, DPCoA
- Constrained methods accept a formula string referencing sample metadata columns
- Returns `skbio.OrdinationResults`

**Plotting** (all return `plotnine.ggplot` objects)
- `plot_bar`, `plot_richness`, `plot_ordination`, `plot_heatmap`, `plot_tree`
- `make_network` / `plot_network` — sample similarity networks

**Hypothesis testing**
- `multi_tax_test` — per-taxon t-test or Wilcoxon rank-sum, with BH, BY, Holm, Bonferroni, or Westfall-Young correction

## Dependencies

| Package | Min version |
|---|---|
| numpy | 1.24 |
| pandas | 2.0 |
| scipy | 1.11 |
| scikit-bio | 0.7 |
| plotnine | 0.13 |
| biom-format | 2.1 |
| h5py | 3.9 |
| pyarrow | 12 |
| pyyaml | 6 |

## Development

```bash
git clone https://github.com/alittleb3ar/pyloseq
cd pyloseq
pip install -e ".[dev]"
pytest
```

Tests use golden files generated from R's phyloseq as numerical ground truth. See [docs/golden_files.md](docs/golden_files.md) for details on regenerating them.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
