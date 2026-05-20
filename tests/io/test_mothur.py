"""Tests for mothur output file I/O."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pyloseq
from pyloseq import Phyloseq


def _write_mothur_shared(path: Path, ps: Phyloseq, cutoff: str = "0.03") -> None:
    df = ps.otu_table.to_dataframe()
    if ps.otu_table.taxa_are_rows:
        df = df.T
    otus = list(df.columns)
    with open(path, "w") as fh:
        fh.write("label\tGroup\tnumOtus\t" + "\t".join(otus) + "\n")
        for sample in df.index:
            vals = "\t".join(str(int(v)) for v in df.loc[sample])
            fh.write(f"{cutoff}\t{sample}\t{len(otus)}\t{vals}\n")


def _write_mothur_constaxonomy(path: Path, ps: Phyloseq) -> None:
    if ps.tax_table is None:
        return
    tax_df = ps.tax_table.to_frame()
    with open(path, "w") as fh:
        fh.write("OTU\tSize\tTaxonomy\n")
        for otu_id in tax_df.index:
            row = tax_df.loc[otu_id]
            taxonomy = ";".join(f"{v}(99)" for v in row if pd.notna(v) and v != "")
            fh.write(f"{otu_id}\t100\t{taxonomy};\n")


def test_mothur_shared_only(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    shared = tmp_path / "test.shared"
    _write_mothur_shared(shared, ps_otu_only)
    ps2 = pyloseq.read_mothur(shared=shared)
    assert ps2.ntaxa == ps_otu_only.ntaxa
    assert ps2.nsamples == ps_otu_only.nsamples


def test_mothur_shared_with_taxonomy(
    ps_with_tax_only: Phyloseq, tmp_path: Path
) -> None:
    shared = tmp_path / "test.shared"
    constax = tmp_path / "test.cons.taxonomy"
    _write_mothur_shared(shared, ps_with_tax_only)
    _write_mothur_constaxonomy(constax, ps_with_tax_only)
    ps2 = pyloseq.read_mothur(shared=shared, constaxonomy=constax)
    assert ps2.tax_table is not None
    assert ps2.ntaxa == ps_with_tax_only.ntaxa


def test_mothur_shared_with_cutoff(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    shared = tmp_path / "multi.shared"
    df = ps_otu_only.otu_table.to_dataframe()
    if ps_otu_only.otu_table.taxa_are_rows:
        df = df.T
    otus = list(df.columns)
    with open(shared, "w") as fh:
        fh.write("label\tGroup\tnumOtus\t" + "\t".join(otus) + "\n")
        for cutoff in ("0.03", "0.05"):
            for sample in df.index:
                vals = "\t".join(str(int(v)) for v in df.loc[sample])
                fh.write(f"{cutoff}\t{sample}\t{len(otus)}\t{vals}\n")
    ps2 = pyloseq.read_mothur(shared=shared, cutoff="0.05")
    assert ps2.ntaxa == ps_otu_only.ntaxa


def test_mothur_show_cutoffs(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    shared = tmp_path / "multi2.shared"
    df = ps_otu_only.otu_table.to_dataframe()
    if ps_otu_only.otu_table.taxa_are_rows:
        df = df.T
    otus = list(df.columns)
    with open(shared, "w") as fh:
        fh.write("label\tGroup\tnumOtus\t" + "\t".join(otus) + "\n")
        for cutoff in ("0.03", "0.05", "0.10"):
            for sample in df.index:
                vals = "\t".join(str(int(v)) for v in df.loc[sample])
                fh.write(f"{cutoff}\t{sample}\t{len(otus)}\t{vals}\n")
    cutoffs = pyloseq.show_mothur_cutoffs(shared)
    assert set(cutoffs) == {"0.03", "0.05", "0.10"}


def test_mothur_no_source_raises() -> None:
    with pytest.raises(ValueError, match="shared"):
        pyloseq.read_mothur()


def test_mothur_select_cutoff_returns_correct_rows(
    ps_otu_only: Phyloseq, tmp_path: Path
) -> None:
    shared = tmp_path / "multi.shared"
    df = ps_otu_only.otu_table.to_dataframe()
    if ps_otu_only.otu_table.taxa_are_rows:
        df = df.T
    otus = list(df.columns)
    with open(shared, "w") as fh:
        fh.write("label\tGroup\tnumOtus\t" + "\t".join(otus) + "\n")
        for cutoff in ("0.03", "0.05"):
            for sample in df.index:
                vals = "\t".join(str(int(v)) for v in df.loc[sample])
                fh.write(f"{cutoff}\t{sample}\t{len(otus)}\t{vals}\n")
    result = pyloseq.select_mothur_cutoff(shared, cutoff="0.05")
    assert isinstance(result, pd.DataFrame)
    # Raw rows: Group column holds sample names, label column holds the cutoff
    assert set(result["Group"]) == set(df.index)
    assert all(c in result.columns for c in otus)
    assert list(result["label"].unique()) == ["0.05"]
