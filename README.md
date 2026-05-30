# pyloseq

**pyloseq** is a native Python port of the R/Bioconductor package [phyloseq](https://joey711.github.io/phyloseq/), built on the PyData stack (pandas, NumPy, scikit-bio, plotnine). It provides first-class QIIME 2 `.qza` support and is designed for researchers migrating 16S/ITS microbiome workflows from R to Python.

Every public function includes an `R reference:` block in its docstring showing the equivalent phyloseq R signature.

## Installation

```bash
pip install pyloseq
```

Requires Python 3.10+.
