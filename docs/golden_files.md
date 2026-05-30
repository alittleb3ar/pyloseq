# Golden Files

Golden files are Parquet (and Newick/FASTA) snapshots of phyloseq's built-in reference
datasets. Tests use them as the ground
truth for numerical comparisons without requiring an R installation at test time.

## Datasets

| Name | Samples | Taxa | Has tree | Has refseq |
|---|---|---|---|---|
| `GlobalPatterns` | 26 | 19,216 | yes | no |
| `enterotype` | 280 | 553 | no | no |
| `esophagus` | 3 | 58 | yes | no |
| `soilrep` | 56 | 16,825 | no | no |

## File layout

```
tests/golden/
├── PROVENANCE.json           # R/package versions + generation timestamp
├── GlobalPatterns/
│   ├── otu_table.parquet
│   ├── sample_data.parquet
│   ├── tax_table.parquet
│   ├── phy_tree.nwk
│   ├── taxa_sums.parquet
│   └── sample_sums.parquet
├── enterotype/
│   ├── otu_table.parquet
│   ├── sample_data.parquet
│   ├── tax_table.parquet
│   ├── taxa_sums.parquet
│   └── sample_sums.parquet
├── esophagus/
│   └── …
└── soilrep/
    └── …
```

## Regenerating golden files

You need R ≥ 4.3 and Bioconductor phyloseq ≥ 1.54. The simplest path is the provided
Docker image:

```bash
docker build -f scripts/Dockerfile.golden -t pyloseq-golden .
docker run --rm -v "$(pwd)/tests/golden:/repo/tests/golden" pyloseq-golden
```

Or directly on a machine with R:

```bash
Rscript scripts/requirements.R   # install deps (once)
Rscript scripts/generate_golden.R
```

## Pinning R package versions

`PROVENANCE.json` records the exact versions used. To reproduce a specific generation:

```r
BiocManager::install("phyloseq", version = "1.54.0")
```

See `scripts/Dockerfile.golden` for the fully pinned Docker image used in CI.
