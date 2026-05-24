# Contributing to phyla

Thank you for contributing! This document covers the development workflow.

## Setup

```bash
git clone https://github.com/mikedonovan/phyla
cd phyla
pip install -e ".[dev]"
pre-commit install
```

## Running checks

```bash
hatch run test         # pytest
hatch run lint         # ruff check
hatch run fmt          # ruff format
hatch run typecheck    # mypy
```

Or directly:

```bash
pytest tests/ -v
ruff check src tests
mypy src/phyla
```

## Pull requests

- One feature / bug fix per PR.
- All CI checks must be green before merge.
- New public functions require a docstring with an `R reference:` block (see existing code for examples).
- Golden-file tests must pass for any function that has a numerical R counterpart. If you add a new analysis function, regenerate the golden files and commit them via Git LFS.

## Regenerating golden files (requires R)

The golden files in `tests/golden/` capture reference outputs from R phyloseq 1.54.x. To regenerate:

1. Install R dependencies:
   ```bash
   Rscript scripts/requirements.R
   ```

2. Run the oracle script:
   ```bash
   Rscript scripts/generate_golden.R
   ```

3. Commit the updated Parquet/Newick/FASTA files (they are tracked via Git LFS).

A Docker image with the pinned R environment is documented in `docs/golden_files.md`.

## Code style

- Line length: 100 characters.
- No comments unless the WHY is non-obvious.
- No docstrings that just restate the function name.
- Type hints required on all public signatures (`mypy --strict`).

## Reporting bugs

Open an issue on GitHub with a minimal reproducible example and the output of:
```python
import phyla; print(phyla.__version__)
import sys; print(sys.version)
```
