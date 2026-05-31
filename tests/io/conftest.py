from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyloseq import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable


def _make_ps(
    n_taxa: int = 5,
    n_samples: int = 3,
    with_sam: bool = True,
    with_tax: bool = True,
    with_tree: bool = False,
) -> Phyloseq:
    rng = np.random.default_rng(42)
    taxa = [f"OTU{i + 1}" for i in range(n_taxa)]
    samples = [f"S{j + 1}" for j in range(n_samples)]

    df = pd.DataFrame(
        rng.integers(0, 200, size=(n_taxa, n_samples)).astype(float),
        index=taxa,
        columns=samples,
    )
    otu = OtuTable(df)

    sam = None
    if with_sam:
        sam = SampleData(
            pd.DataFrame(
                {
                    "group": ["A", "B", "A"][:n_samples],
                    "depth": [1000, 2000, 1500][:n_samples],
                },
                index=samples,
            )
        )

    tax = None
    if with_tax:
        tax = TaxTable(
            pd.DataFrame(
                {
                    "Kingdom": ["Bacteria"] * n_taxa,
                    "Phylum": [f"Phylum{i}" for i in range(n_taxa)],
                },
                index=taxa,
            )
        )

    tree = None
    if with_tree:
        nwk = "(" + ",".join(f"{t}:0.1" for t in taxa) + ");"
        tree = PhyTree.from_newick(nwk)

    return Phyloseq(otu=otu, sam=sam, tax=tax, tree=tree)


@pytest.fixture
def ps_default() -> Phyloseq:
    return _make_ps()


@pytest.fixture
def ps_otu_only() -> Phyloseq:
    return _make_ps(with_sam=False, with_tax=False)


@pytest.fixture
def ps_with_tax_only() -> Phyloseq:
    return _make_ps(with_sam=False, with_tax=True)


@pytest.fixture
def ps_with_sam_only() -> Phyloseq:
    return _make_ps(with_sam=True, with_tax=False)


@pytest.fixture
def ps_with_tree() -> Phyloseq:
    return _make_ps(with_sam=False, with_tax=False, with_tree=True)


@pytest.fixture
def ps_full() -> Phyloseq:
    return _make_ps(with_sam=True, with_tax=True, with_tree=True)
