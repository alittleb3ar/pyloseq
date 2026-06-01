"""Tests for ordination: ordinate() function and Phyloseq.ordinate() method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skbio.stats.ordination import OrdinationResults

import pyloseq
from pyloseq import (OtuTable, Phyloseq, PhyTree, SampleData, TaxTable,
                     distance, ordinate)
from pyloseq._exceptions import pyloseqValidationError
from pyloseq.datasets.fixtures import load_esophagus_reference

GOLDEN_DIR = Path("tests/golden")
ES_GOLDEN = GOLDEN_DIR / "esophagus"
ES_PRESENT = (ES_GOLDEN / "otu_table.parquet").exists()

_NWK = "((OTU1:0.2,OTU2:0.3):0.5,OTU3:0.7);"


def _make_ps(
    *,
    with_sam: bool = True,
    with_tax: bool = True,
    with_tree: bool = False,
) -> Phyloseq:
    otu_data = pd.DataFrame(
        {"S1": [10, 5, 0], "S2": [3, 12, 7], "S3": [0, 1, 20]},
        index=["OTU1", "OTU2", "OTU3"],
    )
    otu = OtuTable(otu_data, taxa_are_rows=True)

    sam = None
    if with_sam:
        sam = SampleData(
            pd.DataFrame(
                {"Group": ["A", "A", "B"], "Depth": [13, 22, 27]},
                index=["S1", "S2", "S3"],
            )
        )

    tax = None
    if with_tax:
        tax = TaxTable(
            pd.DataFrame(
                {
                    "Phylum": ["Firmicutes", "Bacteroidetes", "Proteobacteria"],
                    "Genus": ["Lacto", "Bacter", "Pseudo"],
                },
                index=["OTU1", "OTU2", "OTU3"],
            )
        )

    tree = PhyTree.from_newick(_NWK) if with_tree else None
    return Phyloseq(otu=otu, sam=sam, tax=tax, tree=tree)


@pytest.fixture
def ps() -> Phyloseq:
    return _make_ps()


# ===========================================================================
# ordinate() function
# ===========================================================================


def test_pcoa_returns_ordination(ps: Phyloseq) -> None:

    result = ordinate(ps, method="PCoA", distance="bray")
    assert isinstance(result, OrdinationResults)


def test_mds_alias(ps: Phyloseq) -> None:
    r1 = ordinate(ps, method="PCoA", distance="bray")
    r2 = ordinate(ps, method="MDS", distance="bray")
    np.testing.assert_allclose(
        np.abs(r1.samples.values), np.abs(r2.samples.values), atol=1e-10
    )


def test_pcoa_sample_count(ps: Phyloseq) -> None:
    result = ordinate(ps, method="PCoA", distance="euclidean")
    assert len(result.samples) == ps.nsamples


def test_pcoa_proportion_explained_sums_to_one(ps: Phyloseq) -> None:
    result = ordinate(ps, method="PCoA", distance="euclidean")
    if result.proportion_explained is not None:
        total = result.proportion_explained.dropna().sum()
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_nmds_returns_ordination(ps: Phyloseq) -> None:

    result = ordinate(ps, method="NMDS", distance="bray")
    assert isinstance(result, OrdinationResults)
    assert len(result.samples) == ps.nsamples


def test_unknown_method_raises(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        ordinate(ps, method="GARBAGE")


def test_dca_not_implemented(ps: Phyloseq) -> None:
    with pytest.raises(NotImplementedError):
        ordinate(ps, method="DCA")


def test_ordinate_with_precomputed_dm(ps: Phyloseq) -> None:
    dm = distance(ps, "bray")
    result = ordinate(ps, method="PCoA", distance=dm)
    assert len(result.samples) == ps.nsamples


def test_cca_requires_formula(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        ordinate(ps, method="CCA", formula=None)


def test_rda_with_formula(ps: Phyloseq) -> None:

    result = ordinate(ps, method="RDA", formula="~Group")
    assert isinstance(result, OrdinationResults)


# ===========================================================================
# Phyloseq.ordinate() method
# ===========================================================================


def test_ps_ordinate_returns_ordination_results(ps: Phyloseq) -> None:

    result = ps.ordinate("PCoA", distance="bray")
    assert isinstance(result, OrdinationResults)


def test_ps_ordinate_sample_count(ps: Phyloseq) -> None:
    result = ps.ordinate("PCoA", distance="euclidean")
    assert len(result.samples) == ps.nsamples


def test_ps_ordinate_rda_with_formula(ps: Phyloseq) -> None:

    result = ps.ordinate("RDA", formula="~Group")
    assert isinstance(result, OrdinationResults)


def test_ps_ordinate_unknown_method_raises(ps: Phyloseq) -> None:

    with pytest.raises(pyloseqValidationError, match="Unknown ordination method"):
        ps.ordinate("UMAP")


@pytest.mark.skipif(not ES_PRESENT, reason="golden files not generated yet")
def test_ps_ordinate_pcoa_on_esophagus() -> None:

    ref = load_esophagus_reference()
    ps = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tree=(
            PhyTree.from_newick(ref["phy_tree_newick"])
            if "phy_tree_newick" in ref
            else None
        ),
    )
    result = ps.ordinate("PCoA", distance="bray")
    assert isinstance(result, OrdinationResults)
    assert len(result.samples) == ps.nsamples
