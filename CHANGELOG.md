# Changelog

## [1.0.0] - 2026-05-24

First stable release. Complete Python port of R/Bioconductor phyloseq 1.54.x, validated against R reference outputs via golden-file testing.

- Core containers: `Phyloseq`, `OtuTable`, `SampleData`, `TaxTable`, `PhyTree`, `RefSeq`
- I/O: BIOM v2, QIIME 2 `.qza`, QIIME 1, mothur, CSV
- Manipulation: subset, prune, filter, rarefy, agglomerate, merge, melt
- Analysis: alpha diversity (9 measures), beta diversity (30+ distances), UniFrac, ordination (PCoA, NMDS, MDS, CCA, RDA, CAP)
- Plotting: bar, richness, ordination, heatmap, network (plotnine/ggplot2 API)

## [Unreleased]
