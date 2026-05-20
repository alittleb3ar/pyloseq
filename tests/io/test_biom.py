"""Tests for BIOM v1 (JSON) and v2 (HDF5) I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import pyloseq
from pyloseq import OtuTable, Phyloseq, SampleData, TaxTable
from pyloseq.datasets import load_global_patterns_reference


def _make_gp_ps() -> Phyloseq:
    ref = load_global_patterns_reference()
    return Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
        tax=TaxTable(ref["tax_table"]),
    )


# ===========================================================================
# BIOM v1 (JSON)
# ===========================================================================


def test_biom_v1_write_read_round_trip(ps_default: Phyloseq, tmp_path: Path) -> None:
    p = tmp_path / "test.biom"
    pyloseq.write_biom(ps_default, p, version="1.0")
    ps2 = pyloseq.read_biom(p)
    assert ps2.ntaxa == ps_default.ntaxa
    assert ps2.nsamples == ps_default.nsamples
    assert set(ps2.taxa_names) == set(ps_default.taxa_names)
    assert set(ps2.sample_names) == set(ps_default.sample_names)


def test_biom_v1_taxa_sums_match_r_reference(tmp_path: Path) -> None:
    ref = load_global_patterns_reference()
    ps = _make_gp_ps()
    p = tmp_path / "gp.biom"
    pyloseq.write_biom(ps, p, version="1.0")
    ps2 = pyloseq.read_biom(p)
    rt = ps2.otu_table.taxa_sums().sort_index()
    golden = ref["taxa_sums"].sort_index()
    np.testing.assert_allclose(rt.values, golden.values, atol=1e-6)


def test_biom_v1_taxonomy_cell_values_preserved(tmp_path: Path) -> None:
    ps = _make_gp_ps()
    p = tmp_path / "tax.biom"
    pyloseq.write_biom(ps, p, version="1.0")
    ps2 = pyloseq.read_biom(p)
    assert ps2.tax_table is not None
    tax_df = ps2.tax_table.to_frame()
    assert tax_df.loc["549322", "Kingdom"] == "Archaea"
    assert tax_df.loc["549322", "Phylum"] == "Crenarchaeota"


def test_biom_v1_sample_metadata_values_preserved(tmp_path: Path) -> None:
    ps = _make_gp_ps()
    p = tmp_path / "sam.biom"
    pyloseq.write_biom(ps, p, version="1.0")
    ps2 = pyloseq.read_biom(p)
    assert ps2.sample_data is not None
    sam_df = ps2.sample_data.to_frame()
    assert sam_df.loc["CL3", "SampleType"] == "Soil"


def test_biom_v1_no_taxonomy_parse_none(ps_default: Phyloseq, tmp_path: Path) -> None:
    p = tmp_path / "notax.biom"
    pyloseq.write_biom(ps_default, p, version="1.0")
    ps2 = pyloseq.read_biom(p, parse_taxonomy=None)
    assert ps2.tax_table is None


def test_biom_v1_abundance_values_preserved(
    ps_default: Phyloseq, tmp_path: Path
) -> None:
    p = tmp_path / "vals.biom"
    pyloseq.write_biom(ps_default, p, version="1.0")
    ps2 = pyloseq.read_biom(p)
    orig = ps_default.otu_table.taxa_sums().sort_index()
    rt = ps2.otu_table.taxa_sums().sort_index()
    np.testing.assert_allclose(orig.values, rt.values, atol=1e-6)


# ===========================================================================
# BIOM v2 (HDF5)
# ===========================================================================


def test_biom_v2_write_read_round_trip(ps_default: Phyloseq, tmp_path: Path) -> None:
    p = tmp_path / "test_v2.biom"
    pyloseq.write_biom(ps_default, p, version="2.1")
    ps2 = pyloseq.read_biom(p)
    assert ps2.ntaxa == ps_default.ntaxa
    assert ps2.nsamples == ps_default.nsamples


def test_biom_v2_sparse_input_preserved(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mat = sp.random(50, 10, density=0.1, format="csr", random_state=rng)
    taxa = [f"OTU{i}" for i in range(50)]
    samples = [f"S{j}" for j in range(10)]
    otu = OtuTable(mat, taxa_are_rows=True)
    otu.taxa_names = pd.Index(taxa)
    otu.sample_names = pd.Index(samples)
    ps = Phyloseq(otu=otu)

    p = tmp_path / "sparse.biom"
    pyloseq.write_biom(ps, p, version="2.1")
    ps2 = pyloseq.read_biom(p)
    assert ps2.ntaxa == 50
    assert ps2.nsamples == 10


def test_biom_v2_taxa_sums_match_r_reference(tmp_path: Path) -> None:
    ref = load_global_patterns_reference()
    ps = _make_gp_ps()
    p = tmp_path / "gp_v2.biom"
    pyloseq.write_biom(ps, p, version="2.1")
    ps2 = pyloseq.read_biom(p)
    rt = ps2.otu_table.taxa_sums().sort_index()
    golden = ref["taxa_sums"].sort_index()
    np.testing.assert_allclose(rt.values, golden.values, atol=1e-6)


def test_biom_v2_taxonomy_cell_values_preserved(tmp_path: Path) -> None:
    ps = _make_gp_ps()
    p = tmp_path / "tax_v2.biom"
    pyloseq.write_biom(ps, p, version="2.1")
    ps2 = pyloseq.read_biom(p)
    assert ps2.tax_table is not None
    tax_df = ps2.tax_table.to_frame()
    assert tax_df.loc["549322", "Kingdom"] == "Archaea"


def test_biom_v2_metadata_dict_not_empty(ps_default: Phyloseq, tmp_path: Path) -> None:
    p = tmp_path / "attrs.biom"
    pyloseq.write_biom(ps_default, p, version="2.1")
    ps2 = pyloseq.read_biom(p)
    assert isinstance(ps2.metadata, dict)
    # BIOM v2 HDF5 root attributes must include these standard keys
    assert "generated-by" in ps2.metadata
    assert "creation-date" in ps2.metadata


def test_biom_qiime_vs_default_taxonomy_both_produce_tax_table(
    ps_with_tax_only: Phyloseq, tmp_path: Path
) -> None:
    p = tmp_path / "qiime_tax.biom"
    pyloseq.write_biom(ps_with_tax_only, p, version="2.1")
    ps_default_mode = pyloseq.read_biom(p, parse_taxonomy="default")
    ps_qiime_mode = pyloseq.read_biom(p, parse_taxonomy="qiime")
    assert ps_default_mode.tax_table is not None
    assert ps_qiime_mode.tax_table is not None


def test_biom_v2_values_match_v1(ps_default: Phyloseq, tmp_path: Path) -> None:
    p1 = tmp_path / "v1.biom"
    p2 = tmp_path / "v2.biom"
    pyloseq.write_biom(ps_default, p1, version="1.0")
    pyloseq.write_biom(ps_default, p2, version="2.1")
    ps1 = pyloseq.read_biom(p1)
    ps2 = pyloseq.read_biom(p2)
    s1 = ps1.otu_table.taxa_sums().sort_index()
    s2 = ps2.otu_table.taxa_sums().sort_index()
    np.testing.assert_allclose(s1.values, s2.values, atol=1e-6)
