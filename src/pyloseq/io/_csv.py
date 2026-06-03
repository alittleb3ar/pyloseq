"""Plain CSV/TSV reader and writer.

R reference: phyloseq::phyloseq(otu_table(read.csv(...)), ...)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyloseq._otu_table import OtuTable
from pyloseq._phyloseq import Phyloseq
from pyloseq._refseq import RefSeq
from pyloseq._sample_data import SampleData
from pyloseq._tax_table import TaxTable
from pyloseq._tree import PhyTree

# Index name written by ``to_csv`` to mark the canonical orientation
# (taxa-as-rows). When ``read_csv`` sees it, it knows the file was produced by
# pyloseq and can restore the original orientation regardless of the
# ``taxa_are_rows`` argument.
_TAXA_INDEX_MARKER = "#TAXA"


def read_csv(
    otu_path: str | Path,
    sample_path: str | Path | None = None,
    tax_path: str | Path | None = None,
    tree_path: str | Path | None = None,
    refseq_path: str | Path | None = None,
    taxa_are_rows: bool = True,
    sep: str = "\t",
) -> Phyloseq:
    """Load a plain-text count table (+ optional metadata files) into a Phyloseq.

    R reference: phyloseq::phyloseq(otu_table(read.csv(otu_path), taxa_are_rows), ...)

    Parameters
    ----------
    otu_path:
        Path to the abundance table CSV/TSV.  First column is treated as
        the row index.
    sample_path:
        Optional sample metadata CSV/TSV.  First column is the sample ID
        index.
    tax_path:
        Optional taxonomy CSV/TSV.  First column is the taxon ID index.
    tree_path:
        Optional Newick tree file.
    refseq_path:
        Optional FASTA reference sequences.
    taxa_are_rows:
        Orientation of the OTU table (default ``True``).  Ignored when the
        file carries the pyloseq orientation marker written by :func:`to_csv`
        (such files are always taxa-as-rows).
    sep:
        Field separator (default tab).
    """
    otu_df = pd.read_csv(str(otu_path), sep=sep, index_col=0)

    # If the file was written by ``to_csv`` it is canonically taxa-as-rows,
    # marked by the index name. Trust the marker over the caller's argument so
    # a write/read round-trip never silently transposes the table.
    if otu_df.index.name == _TAXA_INDEX_MARKER:
        taxa_are_rows = True
    otu_df.index.name = None
    otu = OtuTable(otu_df.astype(float), taxa_are_rows=taxa_are_rows)

    sam: SampleData | None = None
    if sample_path is not None:
        sam_df = pd.read_csv(str(sample_path), sep=sep, index_col=0)
        sam_df.index.name = None
        sam = SampleData(sam_df)

    tax: TaxTable | None = None
    if tax_path is not None:
        tax_df = pd.read_csv(str(tax_path), sep=sep, index_col=0)
        tax_df.index.name = None
        tax = TaxTable(tax_df)

    phy_tree: PhyTree | None = None
    if tree_path is not None:
        phy_tree = PhyTree.from_newick_file(Path(tree_path))

    rs: RefSeq | None = None
    if refseq_path is not None:
        rs = RefSeq.from_fasta(Path(refseq_path))

    return Phyloseq(otu=otu, sam=sam, tax=tax, tree=phy_tree, refseq=rs)


def to_csv(
    ps: Phyloseq,
    directory: str | Path,
    sep: str = "\t",
    prefix: str = "",
) -> dict[str, Path]:
    """Write a ``Phyloseq`` to a directory of plain-text files.

    The OTU table is always written in canonical taxa-as-rows orientation and
    tagged with an index marker so :func:`read_csv` restores it faithfully —
    the round-trip is orientation-preserving regardless of how the in-memory
    table happened to be oriented.

    Returns a dict mapping component name → output path.

    R reference: (no direct R equivalent; mirrors write.table() per component)
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _write(df: pd.DataFrame, name: str) -> Path:
        p = out / f"{prefix}{name}.tsv"
        df.to_csv(str(p), sep=sep)
        return p

    # Canonicalize to taxa-as-rows before writing, so orientation is fixed and
    # recoverable rather than dependent on the live object's state.
    otu_df = ps.otu_table.to_dataframe()
    if not ps.otu_table.taxa_are_rows:
        otu_df = otu_df.T
    otu_df = otu_df.copy()
    otu_df.index.name = _TAXA_INDEX_MARKER
    written["otu_table"] = _write(otu_df, "otu_table")

    if ps.sample_data is not None:
        written["sample_data"] = _write(ps.sample_data.to_frame(), "sample_data")

    if ps.tax_table is not None:
        written["tax_table"] = _write(ps.tax_table.to_frame(), "tax_table")

    if ps.phy_tree is not None:
        p = out / f"{prefix}phy_tree.nwk"
        p.write_text(ps.phy_tree.to_newick())
        written["phy_tree"] = p

    if ps.refseq is not None:
        p = out / f"{prefix}refseq.fasta"
        ps.refseq.to_fasta(p)
        written["refseq"] = p

    return written
