# Using pyloseq in Containers

pyloseq has no dedicated Docker image - it doesn't need one. Because it's a pure Python package with standard PyPI dependencies, it installs with a single `pip install` on top of any Python-based base image.

## Quickstart one-liner

```bash
docker run --rm python:3.12-slim sh -c "pip install -q pyloseq && python -c \"
import pyloseq
from pyloseq import Phyloseq, OtuTable, SampleData, TaxTable
from pyloseq.datasets import load_global_patterns_reference
ref = load_global_patterns_reference()
ps = Phyloseq(otu=OtuTable(ref['otu_table'], taxa_are_rows=True), sam=SampleData(ref['sample_data']))
print(ps)
\""
```

## Minimal Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir pyloseq

COPY analysis.py .
CMD ["python", "analysis.py"]
```

## Common base images

All of the following work with a plain `pip install pyloseq`:

| Image | When to use |
|---|---|
| `python:3.12-slim` | Smallest footprint for production pipelines |
| `python:3.12` | Full Debian base; useful when you also need compiled system tools |
| `continuumio/miniconda3` | When the rest of your environment is conda-managed |
| `jupyter/scipy-notebook` | Interactive notebooks with JupyterLab already installed |
| `ubuntu:24.04` | System-level dependencies managed by apt; install Python via `apt-get install python3-pip` first |

## Conda environments

If your base image uses conda, install pyloseq from PyPI into the base or a named environment:

```bash
# into the base conda environment
conda run pip install pyloseq

# or into a named environment
conda create -n microbiome python=3.12
conda run -n microbiome pip install pyloseq
```

pyloseq is not currently on conda-forge. All dependencies that have conda-forge packages will be resolved from there by conda when you later call `conda install`; pyloseq itself is fetched from PyPI.

## Verifying your image

A smoke test to confirm the install is working:

```bash
docker run --rm <your-image> python -c "
import pyloseq
from pyloseq import Phyloseq, OtuTable
print('pyloseq', pyloseq.__version__, 'OK')
"
```

The [`scripts/test_container_install.sh`](https://github.com/alittleb3ar/pyloseq/blob/main/scripts/test_container_install.sh) script in the repository runs this test across all supported base images.
