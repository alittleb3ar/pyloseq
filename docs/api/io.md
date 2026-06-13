# Input / Output

All I/O functions return a `Phyloseq` object (readers) or write files (writers). The I/O module is importable directly from the top-level package:

```python
from pyloseq import read_biom, write_biom, read_qza, read_qiime, read_mothur, read_csv, to_csv
```

---

## BIOM

[BIOM](https://biom-format.org/) is the standard interchange format for microbiome count tables. Both BIOM v1 (JSON) and BIOM v2 (HDF5) are supported.

### read_biom

```python
ps = read_biom("feature-table.biom")
```

Taxonomy parsing is controlled by the `parse_taxonomy` parameter:

| Value | Behaviour |
|---|---|
| `"default"` | Split on `"; "` or `";"`, strip rank prefixes like `"p__"` |
| `"qiime"` | QIIME 1 / GreenGenes semicolon-delimited strings |
| `"greengenes"` | Synonym for `"qiime"` |
| `None` | Store raw taxonomy strings as a single column |
| callable | Called with the raw observation metadata dict; must return a dict of rank → value |

```python
# No taxonomy parsing — keep raw strings
ps = read_biom("table.biom", parse_taxonomy=None)

# Custom parser
def my_parser(obs_meta):
    lineage = obs_meta.get("taxonomy", [])
    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    return dict(zip(ranks, lineage))

ps = read_biom("table.biom", parse_taxonomy=my_parser)
```

::: pyloseq.read_biom

### write_biom

```python
write_biom(ps, "output.biom")
```

Writes a BIOM v2 (HDF5) file by default. The file format version is not configurable; BIOM v2 is universally supported by downstream tools.

::: pyloseq.write_biom

---

## QIIME 2

### read_qza

Reads QIIME 2 `.qza` artifact files without requiring the `qiime2` package. The artifact's semantic type is detected from the embedded `metadata.yaml`:

| Semantic type | Loaded as |
|---|---|
| `FeatureTable[Frequency]` | `OtuTable` |
| `FeatureTable[RelativeFrequency]` | `OtuTable` |
| `FeatureData[Taxonomy]` | `TaxTable` |
| `FeatureData[Sequence]` | `RefSeq` |
| `Phylogeny[Rooted]` | `PhyTree` |
| `Phylogeny[Unrooted]` | `PhyTree` |

Pass multiple `.qza` files to load a complete dataset:

```python
from pyloseq.io import read_qza

ps = read_qza(
    "feature-table.qza",
    "taxonomy.qza",
    "rooted-tree.qza",
)
```

::: pyloseq.io.read_qza

---

## QIIME 1

### read_qiime

Reads QIIME 1 legacy files: a BIOM OTU table and an optional mapping file, tree, and reference sequences.

```python
from pyloseq import read_qiime

ps = read_qiime(
    otu="otu_table.biom",
    mapping="sample_metadata.txt",
    tree="rep_set.tre",
)
```

The `parse_taxonomy` parameter accepts the same values as `read_biom`; the default is `"qiime"` because QIIME 1 uses semicolon-delimited taxonomy strings.

::: pyloseq.read_qiime

---

## mothur

mothur stores results in `.shared` (OTU count table), `.cons.taxonomy` (consensus taxonomy), and `.tre` (tree) files.

### read_mothur

```python
from pyloseq import read_mothur

ps = read_mothur(
    shared="stability.opti_mcc.shared",
    constaxonomy="stability.opti_mcc.0.03.cons.taxonomy",
)
```

mothur shared files often contain multiple OTU definitions at different distance cutoffs. Use `cutoff` to select one:

```python
ps = read_mothur(shared="stability.opti_mcc.shared", cutoff="0.03")
```

Pass a `.list` + `.group` combination instead of a `.shared` file to reconstruct the OTU table from raw assignments:

```python
ps = read_mothur(list_file="stability.list", group="stability.groups", cutoff="0.03")
```

::: pyloseq.read_mothur

### show_mothur_cutoffs

```python
from pyloseq import show_mothur_cutoffs

cutoffs = show_mothur_cutoffs("stability.opti_mcc.shared")
# ['unique', '0.01', '0.02', '0.03']
```

::: pyloseq.show_mothur_cutoffs

### select_mothur_cutoff

Extracts the count table for a single cutoff label. Returns a DataFrame rather than a `Phyloseq` object, useful for inspecting the data before constructing a full object.

::: pyloseq.select_mothur_cutoff

---

## CSV / TSV

For plain-text count tables not in any of the above formats.

### read_csv

```python
from pyloseq import read_csv

ps = read_csv(
    otu_path="otu_table.tsv",
    sample_path="metadata.tsv",
    tax_path="taxonomy.tsv",
    taxa_are_rows=True,
    sep="\t",
)
```

Only `otu_path` is required; other paths are optional. The OTU table index is used as taxa names; the sample table index is used as sample names.

::: pyloseq.read_csv

### to_csv

Writes each component to a separate file. Components not present in the `Phyloseq` are skipped; passing a path for a missing component raises `pyloseqValidationError`.

```python
from pyloseq import to_csv

to_csv(
    ps,
    otu_path="otu_table.tsv",
    sample_path="metadata.tsv",
    tax_path="taxonomy.tsv",
    sep="\t",
)
```

::: pyloseq.to_csv

---

## DESeq2

`Phyloseq.to_deseq2()` exports the count matrix and sample metadata in the format expected by [pydeseq2](https://pydeseq2.readthedocs.io/). It returns a `(counts, metadata)` tuple — both plain `pd.DataFrame` objects — ready to pass directly to `DeseqDataSet`. pydeseq2 is **not** a pyloseq dependency; install it separately with `pip install pydeseq2`.

```python
# pip install pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

counts, metadata = ps.to_deseq2()

dds = DeseqDataSet(counts=counts, metadata=metadata, design="~condition")
dds.deseq2()

ds = DeseqStats(dds, contrast=["condition", "treated", "control"])
ds.summary()
results = ds.results_df
```

`counts` has shape `(n_samples, n_taxa)` with samples as rows. Pass raw, un-normalized integer counts — DESeq2 performs its own size-factor normalization internally. A `UserWarning` is emitted if non-integer values are detected.

`to_deseq2()` raises `ValueError` if `sample_data` is not attached.

::: pyloseq.Phyloseq.to_deseq2
