"""Tests for ordination: ordinate() function and Phyloseq.ordinate() method."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import _make_ps, requires_golden
from skbio.stats.ordination import OrdinationResults

import pyloseq
from pyloseq import OtuTable, Phyloseq, PhyTree, distance, ordinate
from pyloseq._exceptions import pyloseqValidationError
from pyloseq.datasets.fixtures import load_esophagus_reference


@pytest.fixture
def ps_ordination() -> Phyloseq:
    return _make_ps(ntaxa=3, nsamples=3)

# ===========================================================================
# ordinate() — structural invariants across methods
# ===========================================================================

@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("PCoA", {"distance": "bray"}),
        ("NMDS", {"distance": "bray"}),
        ("CA", {}),
    ],
)
def test_ordination_returns_valid_result(
    ps_ordination: Phyloseq, method: str, kwargs: dict[str, Any]
) -> None:
    """Each method returns an OrdinationResults with one row per sample."""
    result = ordinate(ps_ordination, method=method, **kwargs)
    assert isinstance(result, OrdinationResults)
    assert len(result.samples) == ps_ordination.nsamples

def test_mds_alias(ps_ordination: Phyloseq) -> None:
    """MDS is a strict alias for PCoA — numeric scores must match."""
    r1 = ordinate(ps_ordination, method="PCoA", distance="bray")
    r2 = ordinate(ps_ordination, method="MDS", distance="bray")
    np.testing.assert_allclose(
        np.abs(r1.samples.values), np.abs(r2.samples.values), atol=1e-10
    )

def test_pcoa_proportion_explained_sums_to_one(ps_ordination: Phyloseq) -> None:
    result = ordinate(ps_ordination, method="PCoA", distance="euclidean")
    if result.proportion_explained is not None:
        total = result.proportion_explained.dropna().sum()
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

def test_unknown_method_raises(ps_ordination: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        ordinate(ps_ordination, method="GARBAGE")

def test_dca_not_implemented(ps_ordination: Phyloseq) -> None:
    with pytest.raises(NotImplementedError):
        ordinate(ps_ordination, method="DCA")

def test_ordinate_with_precomputed_dm(ps_ordination: Phyloseq) -> None:
    dm = distance(ps_ordination, "bray")
    result = ordinate(ps_ordination, method="PCoA", distance=dm)
    assert len(result.samples) == ps_ordination.nsamples

def test_cca_requires_formula(ps_ordination: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        ordinate(ps_ordination, method="CCA", formula=None)

def test_rda_with_formula(ps_ordination: Phyloseq) -> None:
    result = ordinate(ps_ordination, method="RDA", formula="~Group")
    assert isinstance(result, OrdinationResults)

# ===========================================================================
# Phyloseq.ordinate() method
# ===========================================================================

def test_ps_ordinate_smoke(ps_ordination: Phyloseq) -> None:
    """Method wrapper returns correct type with correct sample count."""
    result = ps_ordination.ordinate("PCoA", distance="bray")
    assert isinstance(result, OrdinationResults)
    assert len(result.samples) == ps_ordination.nsamples

def test_ps_ordinate_rda_with_formula(ps_ordination: Phyloseq) -> None:
    result = ps_ordination.ordinate("RDA", formula="~Group")
    assert isinstance(result, OrdinationResults)

def test_ps_ordinate_unknown_method_raises(ps_ordination: Phyloseq) -> None:
    with pytest.raises(pyloseqValidationError, match="Unknown ordination method"):
        ps_ordination.ordinate("UMAP")

@requires_golden("esophagus", "otu_table.parquet")
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
