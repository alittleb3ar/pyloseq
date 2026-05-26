"""Tests for ordination: ordinate() function and Phyloseq.ordinate() method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyloseq
from pyloseq import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable, distance, ordinate

GOLDEN_DIR = Path("tests/golden")
ES_GOLDEN = GOLDEN_DIR / "esophagus"
ES_PRESENT = (ES_GOLDEN / "otu_table.parquet").exists()


def _make_ps(
    *,
    with_sam: bool = True,
    with_tax: bool = True,
    with_tree: bool = False,
) -> Phyloseq:
    _NWK = "((OTU1:0.2,OTU2:0.3):0.5,OTU3:0.7);"
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


# ===========================================================================
# ordinate() function
# ===========================================================================


class TestOrdinate:
    def test_pcoa_returns_ordination(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ordinate(ps, method="PCoA", distance="bray")
        assert isinstance(result, OrdinationResults)

    def test_mds_alias(self) -> None:
        ps = _make_ps()
        r1 = ordinate(ps, method="PCoA", distance="bray")
        r2 = ordinate(ps, method="MDS", distance="bray")
        np.testing.assert_allclose(np.abs(r1.samples.values), np.abs(r2.samples.values), atol=1e-10)

    def test_pcoa_sample_count(self) -> None:
        ps = _make_ps()
        result = ordinate(ps, method="PCoA", distance="euclidean")
        assert len(result.samples) == ps.nsamples

    def test_pcoa_proportion_explained_sums_to_one(self) -> None:
        ps = _make_ps()
        result = ordinate(ps, method="PCoA", distance="euclidean")
        if result.proportion_explained is not None:
            total = result.proportion_explained.dropna().sum()
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_nmds_returns_ordination(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ordinate(ps, method="NMDS", distance="bray")
        assert isinstance(result, OrdinationResults)
        assert len(result.samples) == ps.nsamples

    def test_unknown_method_raises(self) -> None:
        ps = _make_ps()
        with pytest.raises(pyloseq.pyloseqValidationError):
            ordinate(ps, method="GARBAGE")

    def test_dca_not_implemented(self) -> None:
        ps = _make_ps()
        with pytest.raises(NotImplementedError):
            ordinate(ps, method="DCA")

    def test_ordinate_with_precomputed_dm(self) -> None:
        ps = _make_ps()
        dm = distance(ps, "bray")
        result = ordinate(ps, method="PCoA", distance=dm)
        assert len(result.samples) == ps.nsamples

    def test_cca_requires_formula(self) -> None:
        ps = _make_ps()
        with pytest.raises(pyloseq.pyloseqValidationError):
            ordinate(ps, method="CCA", formula=None)

    def test_rda_with_formula(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ordinate(ps, method="RDA", formula="~Group")
        assert isinstance(result, OrdinationResults)


# ===========================================================================
# Phyloseq.ordinate() method
# ===========================================================================


class TestPsOrdinateMethod:
    def test_returns_ordination_results(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps()
        result = ps.ordinate("PCoA", distance="bray")
        assert isinstance(result, OrdinationResults)

    def test_sample_count(self) -> None:
        ps = _make_ps()
        result = ps.ordinate("PCoA", distance="euclidean")
        assert len(result.samples) == ps.nsamples

    def test_rda_with_formula(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        ps = _make_ps(with_sam=True)
        result = ps.ordinate("RDA", formula="~Group")
        assert isinstance(result, OrdinationResults)

    def test_unknown_method_raises(self) -> None:
        from pyloseq._exceptions import pyloseqValidationError

        ps = _make_ps()
        with pytest.raises(pyloseqValidationError, match="Unknown ordination method"):
            ps.ordinate("UMAP")

    @pytest.mark.skipif(not ES_PRESENT, reason="golden files not generated yet")
    def test_pcoa_on_esophagus(self) -> None:
        from skbio.stats.ordination import OrdinationResults

        from pyloseq.testing.fixtures import load_esophagus_reference

        ref = load_esophagus_reference()
        ps = Phyloseq(
            otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
            tree=PhyTree.from_newick(ref["phy_tree_newick"]) if "phy_tree_newick" in ref else None,
        )
        result = ps.ordinate("PCoA", distance="bray")
        assert isinstance(result, OrdinationResults)
        assert len(result.samples) == ps.nsamples
