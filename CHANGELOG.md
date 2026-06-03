# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-03

Initial release of pyloseq — a native Python port of R/Bioconductor
[phyloseq](https://doi.org/10.1371/journal.pone.0061217) for PyData-native microbiome analysis.

### Data Containers

- `Phyloseq` — main container holding OTU table, sample metadata, taxonomy table, phylogenetic tree, and reference sequences
- `OtuTable` — sparse-aware abundance matrix with automatic CSR storage for sparse data (< 50 % density)
- `SampleData` — per-sample metadata wrapper
- `TaxTable` — taxonomic classification table
- `PhyTree` — phylogenetic tree container backed by scikit-bio
- `RefSeq` — reference sequence container backed by scikit-bio DNA sequences

### I/O

- `read_biom` / `write_biom` — BIOM v1 (JSON) and v2 (HDF5) round-trip
- `read_qiime` — QIIME 1 OTU map and mapping file format
- `read_mothur` — mothur `.shared`, `.list`, `.group`, and `.cons.taxonomy` files
- `read_qza` / `write_qza` — QIIME 2 `.qza` artifact read/write (`FeatureTable[Frequency]`, `FeatureData[Taxonomy]`, `Phylogeny[Rooted]`)
- `read_csv` / `to_csv` — generic CSV/TSV round-trip

### Manipulation

- `filter_taxa`, `prune_taxa`, `prune_samples` — taxon/sample filtering
- `subset_taxa`, `subset_samples` — expression-based subsetting
- `tax_glom`, `tip_glom` — taxonomic and phylogenetic agglomeration
- `merge_phyloseq`, `merge_samples`, `merge_taxa` — combining objects
- `rarefy_even_depth` — rarefaction to even sequencing depth
- `transform_sample_counts` — per-sample count transformation
- `psmelt` — melt Phyloseq to long-format DataFrame
- `kOverA`, `taxa_filter_mask` — abundance filter helpers

### Diversity

- `estimate_richness` — alpha diversity: Observed, Chao1, ACE, Shannon, Simpson, InvSimpson, Fisher
- `distance` / `distance_method_list` — beta diversity distance matrices (all scikit-bio methods)
- `unifrac` — weighted and unweighted UniFrac distances

### Ordination

- `ordinate` — dispatcher supporting PCoA, NMDS, CCA, RDA, DPCoA
- `plot_ordination` — ordination biplots with sample/taxa/split panels

### Plotting

- `plot_bar` — stacked bar charts with taxonomic rank fill
- `plot_richness` — alpha diversity box/point plots
- `plot_heatmap` — abundance heatmaps with ordination-based ordering
- `plot_network` — sample similarity networks (requires networkx)
- `plot_tree` — phylogenetic tree visualization with sample dodging
- `make_network` — network construction helper

### Hypothesis Testing

- `multi_tax_test` — per-taxon two-group tests (t-test, Wilcoxon) with multiple-testing correction: Bonferroni, Holm, BH, BY, Westfall–Young step-down FWER

### Known Limitations

- **DCA ordination** (`method="DCA"`) is not implemented; a `NotImplementedError` with a helpful message is raised.
- **`PhyTree.from_ape_rds()`** is not implemented; use scikit-bio or ete3 to load Newick/Nexus trees directly.
- **`se.ACE`** (standard error of the ACE richness estimator) is not computed; the point estimate (`ACE`) is returned and a warning is issued.
- Nine integration tests are skipped pending generation of R golden files for numerical equivalence validation. See [docs/golden_files.md](docs/golden_files.md) for instructions.

[1.0.0]: https://github.com/alittleb3ar/pyloseq/releases/tag/v1.0.0
