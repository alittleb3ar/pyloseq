# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `GUnifracResult` — new structured return type for `gunifrac()`, exported from `pyloseq`. Supports subscript access (`result["d_0.5"]`), `.keys()`, `.values()`, and `.items()` for drop-in backwards compatibility with the previous `dict` return. Fixed matrices are also accessible as attributes: `result.d_UW`, `result.d_VAW`. **Note:** `isinstance(result, dict)` and `.get()` no longer work — use subscript or attribute access instead.
- `py.typed` marker — pyloseq now declares PEP 561 type information, letting downstream type checkers (mypy, pyright) use the library's annotations without extra configuration.
- `estimate_richness`: new `"PD"` measure — Faith's phylogenetic diversity (Faith 1992). Requires `phy_tree` on the Phyloseq object; the tree is midpoint-rooted internally. Excluded from the default measure set when no tree is present.
- `plot_rarefaction_curve(ps, step, n_steps, color, rng_seed)` — rarefaction curves as a plotnine ggplot. Subsamples each sample at `n_steps` evenly-spaced depths between `step` and the minimum sample depth; optionally colors curves by a `sample_data` column.
- `permanova(distance_matrix, ps, grouping_var, permutations)` — PERMANOVA on a precomputed distance matrix. Thin wrapper around `skbio.stats.distance.permanova` that extracts group labels from `ps.sample_data` and aligns them to the distance matrix IDs automatically. R reference: `vegan::adonis2`.
- `betadisper(distance_matrix, ps, grouping_var, permutations)` — PERMDISP test for homogeneity of multivariate dispersions. Same interface as `permanova`; wraps `skbio.stats.distance.permdisp`. R reference: `vegan::betadisper + vegan::permutest`.

### Added (continued)

- `gunifrac` — Generalized UniFrac distance matrices (Chen et al. 2012 *Bioinformatics* 28:2106–2113). Matches R `GUniFrac` package API.
- `plot_richness`: new `boxplot` parameter (default `True`); set `False` for a points-only plot when boxes add noise to small groups.
- `make_network`: `distance` now accepts a precomputed `skbio.stats.distance.DistanceMatrix` in addition to a metric name string — enables use of custom distances such as those returned by `gunifrac`.
- `plot_network`: edge width is now scaled inversely by distance (closer samples → thicker line), matching R `plot_net`. Shape legends with > 6 unique values are suppressed automatically to avoid a plotnine rendering crash.
- `plot_heatmap` — `method=None` skips ordination and preserves the original sample/taxa order; `label` relabels the x-axis ticks from a `sample_data` column; `taxa_label` relabels the y-axis ticks from a taxonomic rank. These mirror R phyloseq's `method=NULL`, `sample.label`, and `taxa.label`.
- New example notebook: Torondel et al. (2016) pit-latrine microbiome case study demonstrating GUniFrac-based network analysis.

### Changed

- `gunifrac()` return type is now `GUnifracResult` instead of `dict[str, DistanceMatrix]`.
- `SampleData.names` and `RefSeq.names` now emit `DeprecationWarning` on access. Migrate to `.sample_names` and `.taxa_names` respectively.

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

[1.0.0]: https://github.com/alittleb3ar/pyloseq/releases/tag/v1.0.0
