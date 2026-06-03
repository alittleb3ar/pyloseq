"""QIIME 2 ``.qza`` artifact reader and writer.

Does **not** require the ``qiime2`` package at runtime — only ``zipfile``,
``pyyaml``, and ``biom-format``.

R reference: qiime2R::qza_to_phyloseq(...)
"""

from __future__ import annotations

import tempfile
import uuid as _uuid_mod
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pyloseq._phyloseq import Phyloseq
from pyloseq._refseq import RefSeq
from pyloseq._sample_data import SampleData
from pyloseq._tax_table import TaxTable
from pyloseq._tree import PhyTree
from pyloseq.io._biom import (
    _DEFAULT_RANKS,
    _parse_taxonomy_entry,
    read_biom,
    write_biom,
)

# Semantic types we recognise
_FEATURE_TABLE_TYPES = {
    "FeatureTable[Frequency]",
    "FeatureTable[RelativeFrequency]",
    "FeatureTable[Composition]",
}
_TAXONOMY_TYPES = {"FeatureData[Taxonomy]"}
_SEQUENCE_TYPES = {"FeatureData[Sequence]"}
_ROOTED_TREE_TYPES = {"Phylogeny[Rooted]"}
_UNROOTED_TREE_TYPES = {"Phylogeny[Unrooted]"}


def _extract_to_tmp(
    zf: zipfile.ZipFile,
    names: list[str],
    archive_name: str,
    dest: Path,
) -> None:
    """Check that *archive_name* exists in *zf* and write it to *dest*."""
    if archive_name not in names:
        raise ValueError(f"Expected {archive_name} inside {zf.filename}")
    dest.write_bytes(zf.read(archive_name))


def _read_qza_artifact(path: str | Path) -> dict[str, Any]:
    """Extract one .qza artifact and return a dict of pyloseq components."""

    result: dict[str, Any] = {}

    with zipfile.ZipFile(str(path)) as zf:
        names = zf.namelist()

        # metadata.yaml sits one level inside the UUID directory
        meta_paths = [
            n for n in names if n.endswith("/metadata.yaml") and n.count("/") == 1
        ]
        if not meta_paths:
            raise ValueError(f"No metadata.yaml found in {path}")
        meta_path = meta_paths[0]
        uuid = meta_path.split("/")[0]

        with zf.open(meta_path) as fh:
            qza_meta: dict[str, Any] = yaml.safe_load(fh)

        semantic_type: str = qza_meta.get("type", "")
        result["_semantic_type"] = semantic_type
        result["_uuid"] = uuid

        # Stash provenance bytes for round-trip export
        prov: dict[str, bytes] = {}
        for name in names:
            if f"{uuid}/provenance/" in name and not name.endswith("/"):
                prov[name] = zf.read(name)
        result["_provenance"] = prov

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            if semantic_type in _FEATURE_TABLE_TYPES:
                biom_path = tmp / "feature-table.biom"
                _extract_to_tmp(zf, names, f"{uuid}/data/feature-table.biom", biom_path)
                ps = read_biom(biom_path, parse_taxonomy=None)
                result["otu_table"] = ps.otu_table

            elif semantic_type in _TAXONOMY_TYPES:
                tsv_path = tmp / "taxonomy.tsv"
                _extract_to_tmp(zf, names, f"{uuid}/data/taxonomy.tsv", tsv_path)
                result["tax_table"] = _parse_qiime2_taxonomy(tsv_path)

            elif semantic_type in _ROOTED_TREE_TYPES | _UNROOTED_TREE_TYPES:
                nwk_name = f"{uuid}/data/tree.nwk"
                if nwk_name not in names:
                    raise ValueError(f"Expected {nwk_name} inside {path}")
                nwk_text = zf.read(nwk_name).decode()
                result["phy_tree"] = PhyTree.from_newick(nwk_text)

            elif semantic_type in _SEQUENCE_TYPES:
                fa_path = tmp / "sequences.fasta"
                _extract_to_tmp(zf, names, f"{uuid}/data/sequences.fasta", fa_path)
                result["refseq"] = RefSeq.from_fasta(fa_path)

            else:
                raise ValueError(
                    f"Unsupported QIIME 2 semantic type: {semantic_type!r}\n"
                    "Supported: FeatureTable[Frequency|RelativeFrequency|Composition], "
                    "FeatureData[Taxonomy|Sequence], Phylogeny[Rooted|Unrooted]"
                )

    return result


def _parse_qiime2_taxonomy(path: Path) -> Any:
    """Parse a QIIME 2 taxonomy.tsv into a TaxTable."""

    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = None

    taxon_col = next(
        (c for c in df.columns if c.lower() in ("taxon", "taxonomy")), None
    )
    if taxon_col is None:
        raise ValueError(
            f"No 'Taxon' column found in {path}. Columns: {list(df.columns)}"
        )

    parsed: dict[str, dict[str, str]] = {}
    for feat_id, row in df.iterrows():
        parsed[str(feat_id)] = _parse_taxonomy_entry(row[taxon_col], "qiime")

    tax_df = pd.DataFrame.from_dict(parsed, orient="index")
    # Fill missing rank columns
    for rank in _DEFAULT_RANKS:
        if rank not in tax_df.columns:
            tax_df[rank] = pd.NA
    tax_df = tax_df.reindex(columns=_DEFAULT_RANKS)
    return TaxTable(tax_df)


def read_qza(
    features: str | Path | None = None,
    taxonomy: str | Path | None = None,
    tree: str | Path | None = None,
    metadata: str | Path | None = None,
    sequences: str | Path | None = None,
) -> Any:
    """Load one or more QIIME 2 ``.qza`` artifacts into a ``Phyloseq``.

    Each argument should point to a ``.qza`` file of the matching semantic
    type.  ``metadata`` is a sample-metadata TSV (not a ``.qza``).

    R reference: qiime2R::qza_to_phyloseq(features, tree, taxonomy, metadata)
    """

    components: dict[str, Any] = {}
    provenance: dict[str, bytes] = {}

    for arg in (features, taxonomy, tree, sequences):
        if arg is None:
            continue
        art = _read_qza_artifact(arg)
        provenance.update(art.pop("_provenance", {}))
        art.pop("_semantic_type", None)
        art.pop("_uuid", None)
        components.update(art)

    sam = None
    if metadata is not None:
        sam_df = pd.read_csv(
            str(metadata),
            sep="\t",
            index_col=0,
            comment="#",
            skip_blank_lines=True,
        )
        # QIIME 2 metadata files have a second header row of types — drop it
        if sam_df.index[0] == "#q2:types":
            sam_df = sam_df.iloc[1:]
        sam = SampleData(sam_df)

    otu = components.get("otu_table")
    if otu is None:
        raise ValueError("No FeatureTable artifact provided via `features=`")

    return Phyloseq(
        otu=otu,
        sam=sam,
        tax=components.get("tax_table"),
        tree=components.get("phy_tree"),
        refseq=components.get("refseq"),
        metadata={"qza_provenance": provenance} if provenance else {},
    )


def write_qza(
    ps: Any,
    path: str | Path,
    semantic_type: str = "FeatureTable[Frequency]",
) -> None:
    """Write a ``Phyloseq`` component to a QIIME 2 ``.qza`` artifact.

    Only ``FeatureTable[Frequency]`` is fully supported for export;
    other semantic types write their primary data file.

    R reference: (no R equivalent — QIIME 2 native)
    """

    artifact_uuid = str(_uuid_mod.uuid4())
    meta = {
        "uuid": artifact_uuid,
        "type": semantic_type,
        "format": _SEMANTIC_TYPE_FORMAT.get(semantic_type, "BIOMV210DirFmt"),
    }

    with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta_yaml = yaml.dump(meta, default_flow_style=False)
        zf.writestr(f"{artifact_uuid}/metadata.yaml", meta_yaml)

        if semantic_type in _FEATURE_TABLE_TYPES:
            with tempfile.TemporaryDirectory() as tmpdir:
                biom_path = Path(tmpdir) / "feature-table.biom"
                write_biom(ps, biom_path, version="2.1")
                zf.write(biom_path, f"{artifact_uuid}/data/feature-table.biom")

        elif semantic_type in _TAXONOMY_TYPES:
            if ps.tax_table is None:
                raise ValueError("No tax_table to export as FeatureData[Taxonomy]")
            tax_df = ps.tax_table.to_frame().copy()
            tax_df.insert(
                0,
                "Taxon",
                tax_df.apply(
                    lambda r: "; ".join(str(v) for v in r if pd.notna(v)), axis=1
                ),
            )
            zf.writestr(
                f"{artifact_uuid}/data/taxonomy.tsv",
                tax_df[["Taxon"]].to_csv(sep="\t"),
            )

        elif semantic_type in _ROOTED_TREE_TYPES | _UNROOTED_TREE_TYPES:
            if ps.phy_tree is None:
                raise ValueError("No phy_tree to export as Phylogeny artifact")
            zf.writestr(f"{artifact_uuid}/data/tree.nwk", ps.phy_tree.to_newick())

        else:
            raise ValueError(f"Unsupported export semantic type: {semantic_type!r}")


_SEMANTIC_TYPE_FORMAT: dict[str, str] = {
    "FeatureTable[Frequency]": "BIOMV210DirFmt",
    "FeatureTable[RelativeFrequency]": "BIOMV210DirFmt",
    "FeatureTable[Composition]": "BIOMV210DirFmt",
    "FeatureData[Taxonomy]": "TSVTaxonomyDirectoryFormat",
    "Phylogeny[Rooted]": "NewickDirectoryFormat",
    "Phylogeny[Unrooted]": "NewickDirectoryFormat",
}
