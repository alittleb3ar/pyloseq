"""Tests for CSV/TSV round-trip I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pyloseq
from pyloseq import OtuTable, Phyloseq, TaxTable
from pyloseq.datasets import load_global_patterns_reference


def test_csv_otu_only_round_trip(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    pyloseq.to_csv(ps_otu_only, tmp_path / "out")
    ps2 = pyloseq.read_csv(tmp_path / "out" / "otu_table.tsv")
    assert ps2.ntaxa == ps_otu_only.ntaxa
    assert ps2.nsamples == ps_otu_only.nsamples


def test_csv_full_round_trip(ps_full: Phyloseq, tmp_path: Path) -> None:
    out = tmp_path / "full"
    written = pyloseq.to_csv(ps_full, out)
    ps2 = pyloseq.read_csv(
        written["otu_table"],
        sample_path=written["sample_data"],
        tax_path=written["tax_table"],
        tree_path=written["phy_tree"],
    )
    assert ps2.ntaxa == ps_full.ntaxa
    assert ps2.nsamples == ps_full.nsamples
    assert ps2.sample_data is not None
    assert ps2.tax_table is not None
    assert ps2.phy_tree is not None


def test_csv_abundance_values_preserved(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    out = tmp_path / "vals"
    written = pyloseq.to_csv(ps_otu_only, out)
    ps2 = pyloseq.read_csv(written["otu_table"])
    orig = ps_otu_only.otu_table.taxa_sums().sort_index()
    rt = ps2.otu_table.taxa_sums().sort_index()
    np.testing.assert_allclose(orig.values, rt.values, atol=1e-10)


def test_csv_prefix_option(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    pyloseq.to_csv(ps_otu_only, tmp_path / "pfx", prefix="myproject_")
    assert (tmp_path / "pfx" / "myproject_otu_table.tsv").exists()


def test_csv_taxa_are_rows_false(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    ps_otu_only.otu_table.taxa_are_rows = False
    out = tmp_path / "flipped"
    written = pyloseq.to_csv(ps_otu_only, out)
    ps2 = pyloseq.read_csv(written["otu_table"], taxa_are_rows=False)
    assert ps2.ntaxa == ps_otu_only.ntaxa
    assert ps2.nsamples == ps_otu_only.nsamples


def test_csv_taxa_sums_match_r_reference(tmp_path: Path) -> None:
    ref = load_global_patterns_reference()
    ps = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tax=TaxTable(ref["tax_table"]),
    )
    out = tmp_path / "gp"
    written = pyloseq.to_csv(ps, out)
    ps2 = pyloseq.read_csv(written["otu_table"], tax_path=written["tax_table"])
    # pd.read_csv infers numeric taxon IDs as int64; cast both to str before aligning
    rt = ps2.otu_table.taxa_sums()
    golden = ref["taxa_sums"]
    rt_str = pd.Series(rt.values, index=rt.index.astype(str)).sort_index()
    golden_str = pd.Series(golden.values, index=golden.index.astype(str)).sort_index()
    np.testing.assert_allclose(rt_str.to_numpy(), golden_str.to_numpy(), atol=1e-10)
