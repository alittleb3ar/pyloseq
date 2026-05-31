"""Tests for QIIME 1 legacy OTU table I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pyloseq
from pyloseq import OtuTable, Phyloseq
from pyloseq.datasets import load_esophagus_reference


def _write_qiime1_otu_table(path: Path, ps: Phyloseq) -> None:
    df = ps.otu_table.to_dataframe()
    if not ps.otu_table.taxa_are_rows:
        df = df.T
    with open(path, "w") as fh:
        fh.write("# Constructed from biom file\n")
        fh.write("#OTU ID\t" + "\t".join(str(c) for c in df.columns))
        if ps.tax_table is not None:
            fh.write("\ttaxonomy")
        fh.write("\n")
        for otu_id in df.index:
            row = "\t".join(str(int(v)) for v in df.loc[otu_id])
            tax_str = ""
            if ps.tax_table is not None:
                tax_df = ps.tax_table.to_frame()
                if otu_id in tax_df.index:
                    tax_str = "\t" + "; ".join(
                        f"k__{v}" if i == 0 else str(v)
                        for i, v in enumerate(tax_df.loc[otu_id])
                        if pd.notna(v)
                    )
            fh.write(f"{otu_id}\t{row}{tax_str}\n")


def _write_qiime1_mapping(path: Path, ps: Phyloseq) -> None:
    if ps.sample_data is None:
        return
    df = ps.sample_data.to_frame().copy()
    df.index.name = "#SampleID"
    df.to_csv(str(path), sep="\t")


def test_qiime1_otu_table_only(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    otu_path = tmp_path / "otu_table.txt"
    _write_qiime1_otu_table(otu_path, ps_otu_only)
    ps2 = pyloseq.read_qiime(otu_path)
    assert ps2.ntaxa == ps_otu_only.ntaxa
    assert ps2.nsamples == ps_otu_only.nsamples


def test_qiime1_with_taxonomy(ps_with_tax_only: Phyloseq, tmp_path: Path) -> None:
    otu_path = tmp_path / "otu_tax.txt"
    _write_qiime1_otu_table(otu_path, ps_with_tax_only)
    ps2 = pyloseq.read_qiime(otu_path)
    assert ps2.tax_table is not None
    tax_df = ps2.tax_table.to_frame()
    assert tax_df.loc["OTU1", "Kingdom"] == "Bacteria"
    assert tax_df.loc["OTU1", "Phylum"] == "Phylum0"


def test_qiime1_with_mapping(ps_with_sam_only: Phyloseq, tmp_path: Path) -> None:
    otu_path = tmp_path / "otu.txt"
    map_path = tmp_path / "mapping.txt"
    _write_qiime1_otu_table(otu_path, ps_with_sam_only)
    _write_qiime1_mapping(map_path, ps_with_sam_only)
    ps2 = pyloseq.read_qiime(otu_path, mapping=map_path)
    assert ps2.sample_data is not None
    assert ps2.nsamples == ps_with_sam_only.nsamples


def test_qiime1_abundance_values(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    otu_path = tmp_path / "otu_vals.txt"
    _write_qiime1_otu_table(otu_path, ps_otu_only)
    ps2 = pyloseq.read_qiime(otu_path)
    orig = ps_otu_only.otu_table.taxa_sums().sort_index()
    rt = ps2.otu_table.taxa_sums().sort_index()
    np.testing.assert_allclose(orig.values, rt.values, atol=0)


def test_qiime1_esophagus_sample_sums_match_r(tmp_path: Path) -> None:
    ref = load_esophagus_reference()
    ps = Phyloseq(otu=OtuTable(ref["otu_table"], taxa_are_rows=True))
    otu_path = tmp_path / "esoph.txt"
    _write_qiime1_otu_table(otu_path, ps)
    ps2 = pyloseq.read_qiime(otu_path)
    rt = ps2.otu_table.sample_sums().sort_index()
    golden = ref["sample_sums"].sort_index()
    np.testing.assert_allclose(rt.values, golden.values, atol=0)


def test_qiime1_empty_taxonomy_column_no_taxtable(
    ps_otu_only: Phyloseq, tmp_path: Path
) -> None:
    otu_path = tmp_path / "otu_empty_tax.txt"
    df = ps_otu_only.otu_table.to_dataframe()
    with open(otu_path, "w") as fh:
        fh.write("#OTU ID\t" + "\t".join(df.columns) + "\ttaxonomy\n")
        for otu_id in df.index:
            row = "\t".join(str(int(v)) for v in df.loc[otu_id])
            fh.write(f"{otu_id}\t{row}\t\n")
    ps2 = pyloseq.read_qiime(otu_path)
    assert ps2.tax_table is None
