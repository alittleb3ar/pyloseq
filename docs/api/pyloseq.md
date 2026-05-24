# API Reference

`pyloseq` exposes its entire public surface from the top-level namespace — you only
ever need `import pyloseq` or targeted `from pyloseq import ...` imports.

| Section | Contents |
|---|---|
| [Core containers](core.md) | `Phyloseq`, `OtuTable`, `SampleData`, `TaxTable`, `PhyTree`, `RefSeq` |
| [I/O](io.md) | BIOM, QIIME 2 (`.qza`), QIIME 1, mothur, CSV/TSV |
| [Manipulation](manipulation.md) | subset, prune, filter, rarefy, agglomerate, merge, melt |
| [Analysis](analysis.md) | alpha diversity, beta diversity, ordination |
| [Plotting](plotting.md) | bar, richness, ordination, heatmap, network |
| [Exceptions](exceptions.md) | `pyloseqValidationError` |
| [Testing](testing.md) | golden-file loaders for reference datasets |

## Quick-reference: all public names

::: pyloseq
    options:
      members_order: source
      show_root_heading: false
      show_source: false
      show_if_no_docstring: false
      heading_level: 3
