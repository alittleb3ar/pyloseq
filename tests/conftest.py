from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skbio
import skbio.tree

from pyloseq import OtuTable, Phyloseq, PhyTree, RefSeq, SampleData, TaxTable

# ---------------------------------------------------------------------------
# Shared builder used by io fixtures
# ---------------------------------------------------------------------------


def _make_io_ps(
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


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_ps() -> Phyloseq:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.integers(0, 100, size=(3, 2)).astype(float),
        index=[f"OTU{i}" for i in range(3)],
        columns=[f"S{j}" for j in range(2)],
    )
    return Phyloseq(otu=OtuTable(df))


@pytest.fixture
def full_ps() -> Phyloseq:
    df_otu = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["OTU1", "OTU2"],
        columns=["S1", "S2"],
    )
    df_sam = pd.DataFrame(
        {"group": ["A", "B"], "depth": [100, 200]}, index=["S1", "S2"]
    )
    df_tax = pd.DataFrame(
        {"Phylum": ["Firm", "Bact"], "Genus": ["Lacto", "Bact"]},
        index=["OTU1", "OTU2"],
    )
    return Phyloseq(
        otu=OtuTable(df_otu),
        sam=SampleData(df_sam),
        tax=TaxTable(df_tax),
    )


@pytest.fixture
def ps() -> Phyloseq:
    """6×4 Phyloseq (rng=42, Group sample variable) for distance/diversity tests."""
    rng = np.random.default_rng(42)
    counts = rng.integers(1, 100, size=(6, 4)).astype(float)
    taxa = [f"OTU{i + 1}" for i in range(6)]
    samples = [f"S{i + 1}" for i in range(4)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    sam_df = pd.DataFrame({"Group": ["A", "A", "B", "B"]}, index=samples)
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        sam=SampleData(sam_df),
    )


@pytest.fixture
def ps_with_tree() -> Phyloseq:
    """4-taxon, 5-sample Phyloseq with PhyTree for UniFrac/DPCoA tests."""
    newick = "((OTU1:0.1,OTU2:0.2):0.1,(OTU3:0.15,OTU4:0.05):0.2);"
    tree_node = skbio.tree.TreeNode.read(
        StringIO(newick), format="newick", convert_underscores=False
    )
    rng = np.random.default_rng(7)
    counts = rng.integers(1, 200, size=(4, 5)).astype(float)
    taxa = ["OTU1", "OTU2", "OTU3", "OTU4"]
    samples = [f"S{i + 1}" for i in range(5)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        tree=PhyTree(tree_node),
    )


# ---------------------------------------------------------------------------
# IO fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ps_default() -> Phyloseq:
    return _make_io_ps()


@pytest.fixture
def ps_otu_only() -> Phyloseq:
    return _make_io_ps(with_sam=False, with_tax=False)


@pytest.fixture
def ps_with_tax_only() -> Phyloseq:
    return _make_io_ps(with_sam=False, with_tax=True)


@pytest.fixture
def ps_with_sam_only() -> Phyloseq:
    return _make_io_ps(with_sam=True, with_tax=False)


@pytest.fixture
def ps_tree_only() -> Phyloseq:
    return _make_io_ps(with_sam=False, with_tax=False, with_tree=True)


@pytest.fixture
def ps_full() -> Phyloseq:
    return _make_io_ps(with_sam=True, with_tax=True, with_tree=True)


# ---------------------------------------------------------------------------
# Shared helper: available to test modules via `from conftest import ...`
# ---------------------------------------------------------------------------

_GOLDEN_ROOT = Path(__file__).parent / "golden"


def requires_golden(*path_parts: str) -> pytest.MarkDecorator:
    """Return a skipif mark for tests that depend on a generated golden file."""
    p = _GOLDEN_ROOT.joinpath(*path_parts)
    return pytest.mark.skipif(not p.exists(), reason=f"golden file not generated: {p.name}")


def _make_ps(
    ntaxa: int = 6,
    nsamples: int = 4,
    with_sam: bool = True,
    with_tax: bool = True,
    rng: np.random.Generator | None = None,
) -> Phyloseq:
    """Generic Phyloseq builder shared across combine/transform/pruning/ordination tests."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(ntaxa, nsamples)).astype(float)
    taxa = [f"OTU{i + 1}" for i in range(ntaxa)]
    samples = [f"S{i + 1}" for i in range(nsamples)]
    df = pd.DataFrame(counts, index=taxa, columns=samples)
    otu = OtuTable(df, taxa_are_rows=True)

    sam = None
    if with_sam:
        sam_df = pd.DataFrame(
            {
                "Group": ["A", "A", "B", "B"][:nsamples],
                "Depth": [100.0, 200.0, 150.0, 250.0][:nsamples],
            },
            index=samples,
        )
        sam = SampleData(sam_df)

    tax = None
    if with_tax:
        phylum_vals = [
            "Firmicutes",
            "Firmicutes",
            "Bacteroidetes",
            "Proteobacteria",
            "Proteobacteria",
            "Chlamydiae",
        ][:ntaxa]
        genus_vals = [
            "Genus_A",
            "Genus_A",
            "Genus_B",
            "Genus_C",
            "Genus_D",
            "Genus_E",
        ][:ntaxa]
        tax_df = pd.DataFrame(
            {"Phylum": phylum_vals, "Genus": genus_vals},
            index=taxa,
        )
        tax = TaxTable(tax_df)

    return Phyloseq(otu=otu, sam=sam, tax=tax)


def _make_ps_with_tree() -> Phyloseq:
    """4-taxon, 2-sample Phyloseq with binary tree for tip_glom and tree-preservation tests."""
    newick = "((OTU1:0.1,OTU2:0.1):0.05,(OTU3:0.2,OTU4:0.1):0.05);"
    tree_node = skbio.tree.TreeNode.read(
        StringIO(newick), format="newick", convert_underscores=False
    )
    otu = OtuTable(
        pd.DataFrame(
            [[10, 5], [0, 20], [8, 3], [15, 1]],
            index=["OTU1", "OTU2", "OTU3", "OTU4"],
            columns=["S1", "S2"],
            dtype=float,
        ),
        taxa_are_rows=True,
    )
    return Phyloseq(otu=otu, tree=PhyTree(tree_node))


def _make_ps_with_refseq() -> Phyloseq:
    """3-taxon Phyloseq with RefSeq for refseq-preservation tests."""
    df = pd.DataFrame(
        {"S1": [10.0, 5.0, 0.0], "S2": [3.0, 8.0, 2.0]},
        index=["OTU1", "OTU2", "OTU3"],
    )
    rs = RefSeq(
        {
            "OTU1": skbio.DNA("ACGT"),
            "OTU2": skbio.DNA("TTTT"),
            "OTU3": skbio.DNA("GCGC"),
        }
    )
    return Phyloseq(otu=OtuTable(df, taxa_are_rows=True), refseq=rs)
