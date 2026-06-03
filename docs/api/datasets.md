# Datasets

pyloseq ships with four reference datasets drawn from R's phyloseq package. They are stored as Parquet and Newick files under `tests/golden/` and loaded without any R dependency.

The loaders return a plain dict of DataFrames, Series, and strings — not a pre-built `Phyloseq`. This is intentional: it gives you control over which components to include and in what form.

```python
from pyloseq.datasets import (
    load_global_patterns_reference,
    load_enterotype_reference,
    load_esophagus_reference,
    load_soilrep_reference,
    load_golden,
)
```

---

## Reference datasets

| Dataset | Samples | Taxa | Has tree | Has refseq | Dict keys |
|---|---|---|---|---|---|
| `GlobalPatterns` | 26 | 19,216 | yes | no | `otu_table`, `sample_data`, `tax_table`, `phy_tree_newick`, `taxa_sums`, `sample_sums` |
| `enterotype` | 280 | 553 | no | no | `otu_table`, `sample_data`, `tax_table`, `taxa_sums`, `sample_sums` |
| `esophagus` | 3 | 58 | yes | no | `otu_table`, `sample_data`, `tax_table`, `phy_tree_newick`, `taxa_sums`, `sample_sums` |
| `soilrep` | 56 | 16,825 | no | no | `otu_table`, `sample_data`, `tax_table`, `taxa_sums`, `sample_sums` |

### load_global_patterns_reference

26 environmental samples from nine sample types (ocean, soil, freshwater, skin, mock community, etc.), 7 taxonomic ranks. The most commonly used reference dataset for testing ordination and diversity functions:

```python
from pyloseq import Phyloseq, OtuTable, SampleData, TaxTable, PhyTree
from pyloseq.datasets import load_global_patterns_reference

ref = load_global_patterns_reference()

gp = Phyloseq(
    otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
    sam=SampleData(ref["sample_data"]),
    tax=TaxTable(ref["tax_table"]),
    tree=PhyTree.from_newick(ref["phy_tree_newick"]),
)
```

::: pyloseq.datasets.load_global_patterns_reference

### load_enterotype_reference

280 human gut metagenome samples, 553 genera. No tree. Commonly used for enterotype clustering and genus-level analyses:

```python
from pyloseq.datasets import load_enterotype_reference

ref = load_enterotype_reference()
```

::: pyloseq.datasets.load_enterotype_reference

### load_esophagus_reference

3 human esophagus biopsy samples, 58 OTUs, with a phylogenetic tree. The smallest dataset; useful for quick tests requiring a tree:

```python
from pyloseq.datasets import load_esophagus_reference

ref = load_esophagus_reference()
```

::: pyloseq.datasets.load_esophagus_reference

### load_soilrep_reference

56 soil samples from a warming experiment, 16,825 OTUs. No tree. Useful for testing on a larger, sparse table:

```python
from pyloseq.datasets import load_soilrep_reference

ref = load_soilrep_reference()
```

::: pyloseq.datasets.load_soilrep_reference

---

## load_golden

Loads a pre-computed R output for a specific dataset and function. Used in tests to compare pyloseq results against R's reference values:

```python
from pyloseq.datasets import load_golden

r_richness = load_golden("GlobalPatterns", "estimate_richness")
```

The `function` parameter corresponds to the R function name used in `scripts/generate_golden.R`. Pass `**params` matching the exact parameters used when the file was generated to compute the correct file path.

If the file does not exist, `FileNotFoundError` is raised with instructions for regenerating golden files. See the [golden files developer guide](../golden_files.md) for details.

::: pyloseq.datasets.load_golden
