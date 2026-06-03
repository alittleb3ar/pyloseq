"""Tests for QIIME 2 .qza I/O."""

from __future__ import annotations

import uuid as _uuid_mod
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

import pyloseq
from pyloseq import Phyloseq


def _make_feature_table_qza(tmp_path: Path, ps: Phyloseq) -> Path:
    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "feature-table.qza"

    biom_path = tmp_path / "ft.biom"
    pyloseq.write_biom(ps, biom_path, version="2.1")

    meta = {
        "uuid": artifact_uuid,
        "type": "FeatureTable[Frequency]",
        "format": "BIOMV210DirFmt",
    }
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.write(str(biom_path), f"{artifact_uuid}/data/feature-table.biom")
    return qza_path


def _make_taxonomy_qza(tmp_path: Path, ps: Phyloseq) -> Path:
    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "taxonomy.qza"
    if ps.tax_table is None:
        raise ValueError("ps has no tax_table")
    tax_df = ps.tax_table.to_frame()
    taxon_col = tax_df.apply(
        lambda r: "; ".join(
            f"{k[0].lower()}__{v}" for k, v in r.items() if pd.notna(v)
        ),
        axis=1,
    )
    tsv = "Feature ID\tTaxon\n" + "\n".join(
        f"{idx}\t{taxon}" for idx, taxon in taxon_col.items()
    )
    meta = {
        "uuid": artifact_uuid,
        "type": "FeatureData[Taxonomy]",
        "format": "TSVTaxonomyDirectoryFormat",
    }
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.writestr(f"{artifact_uuid}/data/taxonomy.tsv", tsv)
    return qza_path


def _make_tree_qza(tmp_path: Path, ps: Phyloseq) -> Path:
    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "tree.qza"
    if ps.phy_tree is None:
        raise ValueError("ps has no phy_tree")
    meta = {
        "uuid": artifact_uuid,
        "type": "Phylogeny[Rooted]",
        "format": "NewickDirectoryFormat",
    }
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.writestr(f"{artifact_uuid}/data/tree.nwk", ps.phy_tree.to_newick())
    return qza_path


def test_qza_read_feature_table(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    qza = _make_feature_table_qza(tmp_path, ps_otu_only)
    ps2 = pyloseq.read_qza(features=qza)
    assert ps2.ntaxa == ps_otu_only.ntaxa
    assert ps2.nsamples == ps_otu_only.nsamples


def test_qza_read_with_taxonomy(ps_with_tax_only: Phyloseq, tmp_path: Path) -> None:
    ft_qza = _make_feature_table_qza(tmp_path, ps_with_tax_only)
    tax_qza = _make_taxonomy_qza(tmp_path, ps_with_tax_only)
    ps2 = pyloseq.read_qza(features=ft_qza, taxonomy=tax_qza)
    assert ps2.tax_table is not None
    assert ps2.ntaxa == ps_with_tax_only.ntaxa


def test_qza_read_with_tree(ps_tree_only: Phyloseq, tmp_path: Path) -> None:
    ft_qza = _make_feature_table_qza(tmp_path, ps_tree_only)
    tree_qza = _make_tree_qza(tmp_path, ps_tree_only)
    ps2 = pyloseq.read_qza(features=ft_qza, tree=tree_qza)
    assert ps2.phy_tree is not None
    assert ps2.phy_tree.n_tips == ps_tree_only.ntaxa


def test_qza_provenance_stashed_in_metadata(
    ps_otu_only: Phyloseq, tmp_path: Path
) -> None:
    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "prov.qza"
    biom_path = tmp_path / "ft2.biom"
    pyloseq.write_biom(ps_otu_only, biom_path, version="2.1")
    meta = {
        "uuid": artifact_uuid,
        "type": "FeatureTable[Frequency]",
        "format": "BIOMV210DirFmt",
    }
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.write(str(biom_path), f"{artifact_uuid}/data/feature-table.biom")
        zf.writestr(
            f"{artifact_uuid}/provenance/metadata.yaml", yaml.dump({"action": {}})
        )
    ps2 = pyloseq.read_qza(features=qza_path)
    assert "qza_provenance" in ps2.metadata


def test_qza_write_read_round_trip(ps_otu_only: Phyloseq, tmp_path: Path) -> None:
    qza_out = tmp_path / "out.qza"
    pyloseq.write_qza(ps_otu_only, qza_out, semantic_type="FeatureTable[Frequency]")
    ps2 = pyloseq.read_qza(features=qza_out)
    assert ps2.ntaxa == ps_otu_only.ntaxa
    assert ps2.nsamples == ps_otu_only.nsamples


def test_qza_unsupported_semantic_type_raises(tmp_path: Path) -> None:
    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "bad.qza"
    meta = {"uuid": artifact_uuid, "type": "SomeUnknown[Type]", "format": "X"}
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
    with pytest.raises(ValueError, match="Unsupported"):
        from pyloseq.io._qza import _read_qza_artifact

        _read_qza_artifact(qza_path)


def test_qza_no_features_raises() -> None:
    with pytest.raises(ValueError, match="features="):
        pyloseq.read_qza()
