"""QIIME 1 legacy OTU table / mapping file reader.

R reference: phyloseq::import_qiime(otufilename, mapfilename, treefilename, refseqfilename)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pyloseq.io._biom import _parse_taxonomy_entry


def read_qiime(
    otu: str | Path,
    mapping: str | Path | None = None,
    tree: str | Path | None = None,
    refseq: str | Path | None = None,
    parse_taxonomy: str = "qiime",
) -> Any:
    """Load a QIIME 1 OTU table (+ optional mapping, tree, refseq) into a Phyloseq.

    The OTU table must use the standard ``#OTU ID`` header row.  If a
    ``taxonomy`` column is present it is parsed into a ``TaxTable``.

    R reference: phyloseq::import_qiime(otufilename, mapfilename, treefilename, refseqfilename)
    """
    from pyloseq._otu_table import OtuTable
    from pyloseq._phyloseq import Phyloseq
    from pyloseq._refseq import RefSeq
    from pyloseq._sample_data import SampleData
    from pyloseq._tax_table import TaxTable
    from pyloseq._tree import PhyTree

    # ---- OTU table -------------------------------------------------------
    otu_df = _read_qiime1_otu_table(Path(otu))

    # Separate taxonomy column if present
    tax: TaxTable | None = None
    if "taxonomy" in otu_df.columns:
        tax_series = otu_df.pop("taxonomy")
        if parse_taxonomy is not None:
            tax_rows = {
                otu_id: _parse_taxonomy_entry(val, parse_taxonomy)
                for otu_id, val in tax_series.items()
                if pd.notna(val) and val != ""
            }
            if tax_rows:
                tax = TaxTable(pd.DataFrame.from_dict(tax_rows, orient="index"))

    otu_table = OtuTable(otu_df.astype(float), taxa_are_rows=True)

    # ---- Sample mapping --------------------------------------------------
    sam: SampleData | None = None
    if mapping is not None:
        sam_df = _read_qiime1_mapping(Path(mapping))
        sam = SampleData(sam_df)

    # ---- Tree ------------------------------------------------------------
    phy_tree: PhyTree | None = None
    if tree is not None:
        phy_tree = PhyTree.from_newick_file(Path(tree))

    # ---- RefSeq ----------------------------------------------------------
    rs: RefSeq | None = None
    if refseq is not None:
        rs = RefSeq.from_fasta(Path(refseq))

    return Phyloseq(otu=otu_table, sam=sam, tax=tax, tree=phy_tree, refseq=rs)


def _read_qiime1_otu_table(path: Path) -> pd.DataFrame:
    """Parse a QIIME 1 OTU table text file.

    Handles the ``# Constructed from biom file`` comment header and the
    ``#OTU ID`` index column convention.
    """
    # Skip lines starting with "# " (comments) but keep the "#OTU ID" header
    lines: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("# ") and not line.startswith("#OTU"):
                continue
            lines.append(line)

    import io

    raw = "".join(lines)
    df = pd.read_csv(io.StringIO(raw), sep="\t", index_col=0)

    # Normalise the index name left by "#OTU ID"
    df.index.name = None
    # Strip leading "#" from column names caused by some exporters
    df.columns = [c.lstrip("#").strip() if isinstance(c, str) else c for c in df.columns]
    return df


def _read_qiime1_mapping(path: Path) -> pd.DataFrame:
    """Parse a QIIME 1 sample mapping file.

    The first column is ``#SampleID``; subsequent columns are metadata
    variables.  Trailing ``Description`` column is kept.
    """
    df = pd.read_csv(str(path), sep="\t", index_col=0)
    # The header line begins with "#SampleID" — pandas reads "#SampleID" as
    # the index name.  Normalise it.
    df.index.name = None
    # Drop blank lines that some exporters add
    df = df.dropna(how="all")
    return df
