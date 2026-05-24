"""mothur output file reader.

Supports ``.shared``, ``.cons.taxonomy``, ``.tre``, ``.group``, and
``.list`` files.

R reference: phyloseq::import_mothur(mothur_list_file, mothur_group_file,
             mothur_tree_file, cutoff, mothur_shared_file, mothur_constaxonomy_file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_mothur(
    shared: str | Path | None = None,
    constaxonomy: str | Path | None = None,
    tree: str | Path | None = None,
    list_file: str | Path | None = None,
    group: str | Path | None = None,
    cutoff: str | None = None,
) -> Any:
    """Load mothur output files into a ``Phyloseq``.

    Parameters
    ----------
    shared:
        Path to a ``.shared`` file.  Produces the OTU table.
    constaxonomy:
        Path to a ``.cons.taxonomy`` file.  Produces the ``TaxTable``.
    tree:
        Path to a Newick ``.tre`` file.  Produces the ``PhyTree``.
    list_file:
        Path to a ``.list`` file (alternative to ``shared``).
    group:
        Path to a ``.group`` file (required with ``list_file``).
    cutoff:
        OTU similarity cutoff label (e.g. ``"0.03"``).  If ``None``,
        the first label in the file is used.

    R reference: phyloseq::import_mothur(...)
    """
    from pyloseq._otu_table import OtuTable
    from pyloseq._phyloseq import Phyloseq
    from pyloseq._tax_table import TaxTable
    from pyloseq._tree import PhyTree

    otu_table: OtuTable | None = None
    tax: TaxTable | None = None
    phy_tree: PhyTree | None = None

    # ---- OTU table from .shared -----------------------------------------
    if shared is not None:
        otu_df = _read_shared(Path(shared), cutoff=cutoff)
        otu_table = OtuTable(
            otu_df.astype(float), taxa_are_rows=False
        )  # samples as rows in .shared

    # ---- OTU table from .list + .group ----------------------------------
    elif list_file is not None and group is not None:
        otu_df = _list_group_to_otu(Path(list_file), Path(group), cutoff=cutoff)
        otu_table = OtuTable(otu_df.astype(float), taxa_are_rows=False)

    if otu_table is None:
        raise ValueError("Provide either `shared` or both `list_file` and `group`")

    # ---- Taxonomy --------------------------------------------------------
    if constaxonomy is not None:
        tax_df = _read_constaxonomy(Path(constaxonomy))
        tax = TaxTable(tax_df)

    # ---- Tree ------------------------------------------------------------
    if tree is not None:
        phy_tree = PhyTree.from_newick_file(Path(tree))

    return Phyloseq(otu=otu_table, tax=tax, tree=phy_tree)


def show_mothur_cutoffs(path: str | Path) -> list[str]:
    """Return all OTU cutoff labels present in a ``.list`` or ``.shared`` file.

    R reference: show_mothur_cutoffs(mothurlist)
    """
    cutoffs: list[str] = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0] not in ("label", ""):
                label = parts[0]
                if label not in cutoffs:
                    cutoffs.append(label)
    return cutoffs


def select_mothur_cutoff(path: str | Path, cutoff: str) -> pd.DataFrame:
    """Return the rows of a ``.list`` or ``.shared`` file at a given cutoff.

    R reference: (internal helper, mirrors mothur's label-filtering)
    """
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0] == cutoff:
                rows.append(parts)
    if not rows:
        raise ValueError(f"Cutoff {cutoff!r} not found in {path}")
    return pd.DataFrame(rows, columns=header)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_shared(path: Path, cutoff: str | None) -> pd.DataFrame:
    """Parse a .shared file; returns samples-as-rows OTU count DataFrame."""
    df = pd.read_csv(str(path), sep="\t", dtype=str)
    if cutoff is None:
        cutoff = df["label"].iloc[0]
    df = df[df["label"] == cutoff].copy()
    df = df.drop(columns=["label", "numOtus"], errors="ignore")
    df = df.set_index("Group")
    df.index.name = None
    return df.astype(float)


def _list_group_to_otu(list_path: Path, group_path: Path, cutoff: str | None) -> pd.DataFrame:
    """Convert .list + .group to a samples-as-rows OTU count DataFrame."""
    # group file: seq_name <tab> sample_name
    group_df = pd.read_csv(str(group_path), sep="\t", header=None, names=["seq", "sample"])
    seq_to_sample: dict[str, str] = dict(zip(group_df["seq"], group_df["sample"], strict=False))
    samples = sorted(set(seq_to_sample.values()))

    # list file: label <tab> numOTUs <tab> otu1_seqs <tab> otu2_seqs ...
    with open(list_path) as fh:
        fh.readline()  # skip header: label numOtus Otu001 ...
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if cutoff is None or parts[0] == cutoff:
                otu_cols = parts[2:]  # each is comma-separated seq names
                break
        else:
            raise ValueError(f"Cutoff {cutoff!r} not found in {list_path}")

    otu_names = [f"Otu{i + 1:03d}" for i in range(len(otu_cols))]
    counts: dict[str, dict[str, int]] = {s: dict.fromkeys(otu_names, 0) for s in samples}
    for otu_name, seq_csv in zip(otu_names, otu_cols, strict=False):
        for seq in seq_csv.split(","):
            seq = seq.strip()
            if seq in seq_to_sample:
                counts[seq_to_sample[seq]][otu_name] += 1

    return pd.DataFrame.from_dict(counts, orient="index")[otu_names]


def _read_constaxonomy(path: Path) -> pd.DataFrame:
    """Parse a .cons.taxonomy file into a TaxTable DataFrame."""
    df = pd.read_csv(str(path), sep="\t", index_col=0)
    df.index.name = None

    # Drop Size column if present
    df = df.drop(columns=["Size"], errors="ignore")

    tax_col = next((c for c in df.columns if c.lower() == "taxonomy"), None)
    if tax_col is None:
        raise ValueError(f"No 'Taxonomy' column in {path}. Columns: {list(df.columns)}")

    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    parsed: dict[str, list[str]] = {}
    for otu_id, row in df.iterrows():
        raw: str = str(row[tax_col])
        # mothur format: "Bacteria(100);Firmicutes(90);...;"
        parts = [p.split("(")[0].strip() for p in raw.rstrip(";").split(";") if p.strip()]
        # Pad to rank length
        parts += [""] * max(0, len(ranks) - len(parts))
        parsed[str(otu_id)] = parts[: len(ranks)]

    return pd.DataFrame.from_dict(parsed, orient="index", columns=ranks)
