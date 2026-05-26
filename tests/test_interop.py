"""Tests for AnnData round-trip, PERMANOVA convenience, and Phyloseq method aliases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyloseq import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable
from pyloseq._refseq import RefSeq

GOLDEN_DIR = Path("tests/golden")
GP_PRESENT = (GOLDEN_DIR / "GlobalPatterns" / "otu_table.parquet").exists()
ES_PRESENT = (GOLDEN_DIR / "esophagus" / "otu_table.parquet").exists()

_NWK = "((OTU1:0.2,OTU2:0.3):0.5,OTU3:0.7);"

anndata = pytest.importorskip("anndata", reason="anndata not installed")


def _make_ps(
    *,
    with_sam: bool = True,
    with_tax: bool = True,
    with_tree: bool = False,
    with_refseq: bool = False,
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

    refseq = None
    if with_refseq:
        import skbio

        refseq = RefSeq(
            {
                "OTU1": skbio.DNA("ACGT"),
                "OTU2": skbio.DNA("TTGC"),
                "OTU3": skbio.DNA("GGCA"),
            }
        )

    return Phyloseq(otu=otu, sam=sam, tax=tax, tree=tree, refseq=refseq)


# ===========================================================================
# AnnData round-trip
# ===========================================================================


class TestToAnnData:
    def test_returns_anndata(self) -> None:
        ps = _make_ps()
        ad = ps.to_anndata()
        assert isinstance(ad, anndata.AnnData)

    def test_x_shape_samples_by_taxa(self) -> None:
        ps = _make_ps()
        ad = ps.to_anndata()
        assert ad.X.shape == (ps.nsamples, ps.ntaxa)

    def test_obs_index_matches_sample_names(self) -> None:
        ps = _make_ps()
        ad = ps.to_anndata()
        assert list(ad.obs_names) == list(ps.sample_names)

    def test_var_index_matches_taxa_names(self) -> None:
        ps = _make_ps()
        ad = ps.to_anndata()
        assert list(ad.var_names) == list(ps.taxa_names)

    def test_obs_columns_from_sample_data(self) -> None:
        ps = _make_ps(with_sam=True)
        ad = ps.to_anndata()
        assert "Group" in ad.obs.columns
        assert "Depth" in ad.obs.columns

    def test_var_columns_from_tax_table(self) -> None:
        ps = _make_ps(with_tax=True)
        ad = ps.to_anndata()
        assert "Phylum" in ad.var.columns
        assert "Genus" in ad.var.columns

    def test_obs_empty_when_no_sample_data(self) -> None:
        ps = _make_ps(with_sam=False)
        ad = ps.to_anndata()
        assert ad.obs.shape == (ps.nsamples, 0)

    def test_var_empty_when_no_tax_table(self) -> None:
        ps = _make_ps(with_tax=False)
        ad = ps.to_anndata()
        assert ad.var.shape == (ps.ntaxa, 0)

    def test_uns_tree_newick_present(self) -> None:
        ps = _make_ps(with_tree=True)
        ad = ps.to_anndata()
        assert "phy_tree" in ad.uns
        assert isinstance(ad.uns["phy_tree"], str)
        assert len(ad.uns["phy_tree"]) > 5

    def test_uns_tree_absent_when_no_tree(self) -> None:
        ps = _make_ps(with_tree=False)
        ad = ps.to_anndata()
        assert "phy_tree" not in ad.uns

    def test_uns_refseq_present(self) -> None:
        ps = _make_ps(with_refseq=True)
        ad = ps.to_anndata()
        assert "refseq" in ad.uns
        assert ad.uns["refseq"]["OTU1"] == "ACGT"

    def test_x_values_correct(self) -> None:
        ps = _make_ps()
        ad = ps.to_anndata()
        s1_idx = list(ad.obs_names).index("S1")
        otu1_idx = list(ad.var_names).index("OTU1")
        assert ad.X[s1_idx, otu1_idx] == pytest.approx(10.0)


class TestFromAnnData:
    def test_round_trip_otu_values(self) -> None:
        ps = _make_ps()
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        orig = ps.otu_table.to_dataframe()
        rt = ps2.otu_table.to_dataframe()
        if not ps2.otu_table.taxa_are_rows:
            rt = rt.T
        pd.testing.assert_frame_equal(
            orig.reindex(sorted(orig.index), axis=0)
            .reindex(sorted(orig.columns), axis=1)
            .astype(float),
            rt.reindex(sorted(rt.index), axis=0).reindex(sorted(rt.columns), axis=1).astype(float),
        )

    def test_round_trip_sample_data(self) -> None:
        ps = _make_ps(with_sam=True)
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.sample_data is not None
        assert set(ps2.sample_data.variables) == {"Group", "Depth"}

    def test_round_trip_tax_table(self) -> None:
        ps = _make_ps(with_tax=True)
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.tax_table is not None
        assert set(ps2.tax_table.rank_names) == {"Phylum", "Genus"}

    def test_round_trip_tree(self) -> None:
        ps = _make_ps(with_tree=True)
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.phy_tree is not None
        assert ps2.phy_tree.n_tips == ps.phy_tree.n_tips  # type: ignore[union-attr]

    def test_round_trip_refseq(self) -> None:
        ps = _make_ps(with_refseq=True)
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.refseq is not None
        assert str(ps2.refseq["OTU1"]) == "ACGT"

    def test_round_trip_no_sam_no_tax(self) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.sample_data is None
        assert ps2.tax_table is None
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_ntaxa_nsamples_preserved(self) -> None:
        ps = _make_ps()
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    @pytest.mark.skipif(not GP_PRESENT, reason="golden files not generated yet")
    def test_round_trip_globalpatterns(self) -> None:
        from pyloseq.testing.fixtures import load_global_patterns_reference

        ref = load_global_patterns_reference()
        ps = Phyloseq(
            otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
            sam=SampleData(ref["sample_data"]),
            tax=TaxTable(ref["tax_table"]),
            tree=PhyTree.from_newick(ref["phy_tree_newick"]),
        )
        ps2 = Phyloseq.from_anndata(ps.to_anndata())
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples
        assert ps2.phy_tree is not None
        assert ps2.tax_table is not None
        assert ps2.sample_data is not None


# ===========================================================================
# Phyloseq.distance() convenience method
# ===========================================================================


class TestPsDistanceMethod:
    def test_returns_distance_matrix(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps()
        dm = ps.distance("bray")
        assert isinstance(dm, DistanceMatrix)

    def test_ids_match_sample_names(self) -> None:
        ps = _make_ps()
        dm = ps.distance("euclidean")
        assert list(dm.ids) == list(ps.sample_names)

    def test_symmetric(self) -> None:
        ps = _make_ps()
        dm = ps.distance("bray")
        arr = np.array(dm.data)
        np.testing.assert_allclose(arr, arr.T)

    def test_zero_diagonal(self) -> None:
        ps = _make_ps()
        dm = ps.distance("jaccard")
        np.testing.assert_allclose(np.diag(dm.data), 0.0)

    def test_plugs_into_permanova(self) -> None:
        from skbio.stats.distance import permanova

        ps = _make_ps(with_sam=True)
        dm = ps.distance("bray")
        grouping = ps.sample_data.to_frame()["Group"]  # type: ignore[union-attr]
        result = permanova(dm, grouping)
        assert "p-value" in result.index

    def test_unifrac_requires_tree(self) -> None:
        from pyloseq._exceptions import pyloseqValidationError

        ps = _make_ps(with_tree=False)
        with pytest.raises(pyloseqValidationError, match="phy_tree"):
            ps.distance("unifrac")

    def test_unifrac_with_tree(self) -> None:
        from skbio.stats.distance import DistanceMatrix

        ps = _make_ps(with_tree=True)
        dm = ps.distance("unifrac")
        assert isinstance(dm, DistanceMatrix)


# ===========================================================================
# One-liner interop smoke tests
# ===========================================================================


def test_permanova_one_liner() -> None:
    from skbio.stats.distance import permanova

    ps = _make_ps(with_sam=True)
    result = permanova(ps.distance("bray"), ps.sample_data.to_frame()["Group"])  # type: ignore[union-attr]
    assert "test statistic" in result.index


def test_phase5_methods_on_phyloseq() -> None:
    assert hasattr(Phyloseq, "to_anndata")
    assert hasattr(Phyloseq, "from_anndata")
    assert hasattr(Phyloseq, "distance")
    assert hasattr(Phyloseq, "ordinate")
