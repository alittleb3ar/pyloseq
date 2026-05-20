# Containers

pyloseq represents microbiome data through a small set of typed container classes. `Phyloseq` is the top-level object that holds the others. All containers are immutable in the sense that manipulation functions never modify them in-place — they always return new objects.

---

## Phyloseq

`Phyloseq` is the central data object. It bundles an OTU table with any combination of sample metadata, taxonomic annotations, a phylogenetic tree, and reference sequences. The constructor validates component consistency and silently prunes to the intersection of taxa and sample names across all attached components.

```python
from pyloseq import Phyloseq, OtuTable, SampleData, TaxTable, PhyTree

ps = Phyloseq(
    otu=OtuTable(df, taxa_are_rows=True),
    sam=SampleData(metadata_df),
    tax=TaxTable(taxonomy_df),
    tree=PhyTree.from_newick(newick_str),
)
```

::: pyloseq.Phyloseq

### Validation and pruning

When taxa or sample names differ between components, the constructor prunes each to the intersection and emits a warning. Pass `quiet=True` to suppress the warning. Pass `strict=True` to raise `pyloseqValidationError` instead of pruning:

```python
# Raises if otu and tax have mismatched taxa names
ps = Phyloseq(otu=otu, tax=tax, strict=True)

# Prune silently
ps = Phyloseq(otu=otu, tax=tax, quiet=True)
```

Component setters (e.g., `ps.tax_table = new_tax`) trigger re-validation using the same logic.

---

## OtuTable

Stores the feature abundance matrix. Rows can be taxa or samples — track orientation with `taxa_are_rows`.

```python
import pandas as pd
from pyloseq import OtuTable

df = pd.DataFrame(
    {"S1": [10, 0, 5], "S2": [0, 3, 7]},
    index=["OTU1", "OTU2", "OTU3"],
)
otu = OtuTable(df, taxa_are_rows=True)
```

Sparse input (NumPy sparse matrices, scipy CSR/CSC) is accepted. When the matrix density is below 50%, the internal representation is automatically converted to CSR format.

`to_dataframe()` always returns a DataFrame with **taxa as rows**, regardless of internal orientation:

```python
df = otu.to_dataframe()   # taxa × samples
```

Flip orientation without copying data:

```python
otu.taxa_are_rows = False  # now samples are rows internally
```

::: pyloseq.OtuTable

---

## SampleData

Wraps per-sample metadata as a `pandas.DataFrame`. The DataFrame index is the sample identifier — it must be unique and must match sample names in the OTU table.

```python
import pandas as pd
from pyloseq import SampleData

meta = pd.DataFrame(
    {"SampleType": ["Soil", "Ocean", "Skin"], "pH": [6.5, 8.1, 5.4]},
    index=["S1", "S2", "S3"],
)
sam = SampleData(meta)
```

Retrieve the underlying DataFrame with `.to_frame()`.

::: pyloseq.SampleData

---

## TaxTable

Wraps the taxonomic classification table. Rows are taxa (indexed by the same names as the OTU table rows), columns are taxonomic ranks.

```python
import pandas as pd
from pyloseq import TaxTable

tax_df = pd.DataFrame(
    {
        "Kingdom": ["Bacteria", "Bacteria"],
        "Phylum":  ["Firmicutes", "Bacteroidetes"],
        "Genus":   ["Lactobacillus", "Bacteroides"],
    },
    index=["OTU1", "OTU2"],
)
tax = TaxTable(tax_df)
print(tax.rank_names)   # ['Kingdom', 'Phylum', 'Genus']
```

::: pyloseq.TaxTable

---

## PhyTree

Wraps a `skbio.TreeNode` phylogenetic tree. Three constructors are available:

```python
from pyloseq import PhyTree

# From a Newick string
tree = PhyTree.from_newick("((OTU1:0.1, OTU2:0.2):0.3, OTU3:0.4);")

# From a file
tree = PhyTree.from_newick_file("tree.nwk")

# From an existing skbio.TreeNode
import skbio
node = skbio.io.read("tree.nwk", format="newick", into=skbio.TreeNode)
tree = PhyTree(node)
```

!!! note
    `PhyTree.from_ape_rds()` is not implemented. To use a tree saved from R with `saveRDS`, export it first: `ape::write.tree(tree, "tree.nwk")`.

Prune to a specific set of tips:

```python
pruned = tree.prune(["OTU1", "OTU3"])
```

::: pyloseq.PhyTree

---

## RefSeq

Stores reference sequences as a dict-like mapping from taxon name to `skbio.DNA`. Used for representative sequences from DADA2 or QIIME 2 denoising pipelines.

```python
import skbio
from pyloseq import RefSeq

seqs = RefSeq({
    "OTU1": skbio.DNA("ACGTACGT"),
    "OTU2": skbio.DNA("TGCATGCA"),
})

# Round-trip through FASTA
seqs.to_fasta("representatives.fasta")
seqs2 = RefSeq.from_fasta("representatives.fasta")
```

::: pyloseq.RefSeq
