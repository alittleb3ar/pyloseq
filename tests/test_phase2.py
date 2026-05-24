"""Phase 2 tests: I/O readers and writers.

All test fixtures are created programmatically in tmp_path — no binary
files committed to the repo.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyloseq
from pyloseq import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable

# ===========================================================================
# Shared fixture factory
# ===========================================================================

def _make_ps(
    n_taxa: int = 5,
    n_samples: int = 3,
    with_sam: bool = True,
    with_tax: bool = True,
    with_tree: bool = False,
) -> Phyloseq:
    rng = np.random.default_rng(42)
    taxa = [f"OTU{i+1}" for i in range(n_taxa)]
    samples = [f"S{j+1}" for j in range(n_samples)]

    df = pd.DataFrame(
        rng.integers(0, 200, size=(n_taxa, n_samples)).astype(float),
        index=taxa, columns=samples,
    )
    otu = OtuTable(df)

    sam = None
    if with_sam:
        sam = SampleData(pd.DataFrame(
            {"group": ["A", "B", "A"][:n_samples], "depth": [1000, 2000, 1500][:n_samples]},
            index=samples,
        ))

    tax = None
    if with_tax:
        tax = TaxTable(pd.DataFrame(
            {"Kingdom": ["Bacteria"] * n_taxa, "Phylum": [f"Phylum{i}" for i in range(n_taxa)]},
            index=taxa,
        ))

    tree = None
    if with_tree:
        nwk = "(" + ",".join(f"{t}:0.1" for t in taxa) + ");"
        tree = PhyTree.from_newick(nwk)

    return Phyloseq(otu=otu, sam=sam, tax=tax, tree=tree)


# ===========================================================================
# Ticket 2.1 — BIOM v1 (JSON)
# ===========================================================================

class TestBiomV1:
    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        ps = _make_ps()
        p = tmp_path / "test.biom"
        pyloseq.write_biom(ps, p, version="1.0")
        ps2 = pyloseq.read_biom(p)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples
        assert set(ps2.taxa_names) == set(ps.taxa_names)
        assert set(ps2.sample_names) == set(ps.sample_names)

    def test_taxonomy_preserved(self, tmp_path: Path) -> None:
        ps = _make_ps(with_tax=True)
        p = tmp_path / "tax.biom"
        pyloseq.write_biom(ps, p, version="1.0")
        ps2 = pyloseq.read_biom(p, parse_taxonomy="default")
        assert ps2.tax_table is not None
        assert ps2.ntaxa == ps.ntaxa

    def test_sample_metadata_preserved(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=True)
        p = tmp_path / "sam.biom"
        pyloseq.write_biom(ps, p, version="1.0")
        ps2 = pyloseq.read_biom(p)
        assert ps2.sample_data is not None
        assert set(ps2.sample_data.variables) == {"group", "depth"}

    def test_no_taxonomy_parse_none(self, tmp_path: Path) -> None:
        ps = _make_ps(with_tax=True)
        p = tmp_path / "notax.biom"
        pyloseq.write_biom(ps, p, version="1.0")
        ps2 = pyloseq.read_biom(p, parse_taxonomy=None)
        assert ps2.tax_table is None

    def test_abundance_values_preserved(self, tmp_path: Path) -> None:
        ps = _make_ps()
        p = tmp_path / "vals.biom"
        pyloseq.write_biom(ps, p, version="1.0")
        ps2 = pyloseq.read_biom(p)
        orig = ps.otu_table.taxa_sums().sort_index()
        rt = ps2.otu_table.taxa_sums().sort_index()
        np.testing.assert_allclose(orig.values, rt.values, atol=1e-6)


# ===========================================================================
# Ticket 2.2 — BIOM v2 (HDF5)
# ===========================================================================

class TestBiomV2:
    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        ps = _make_ps()
        p = tmp_path / "test_v2.biom"
        pyloseq.write_biom(ps, p, version="2.1")
        ps2 = pyloseq.read_biom(p)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_sparse_input_preserved(self, tmp_path: Path) -> None:
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

    def test_hdf5_attributes_in_metadata(self, tmp_path: Path) -> None:
        ps = _make_ps()
        p = tmp_path / "attrs.biom"
        pyloseq.write_biom(ps, p, version="2.1")
        ps2 = pyloseq.read_biom(p)
        # HDF5 attrs written by biom-format include creation-date, etc.
        assert isinstance(ps2.metadata, dict)

    def test_taxonomy_qiime_mode(self, tmp_path: Path) -> None:
        """Taxonomy written as list round-trips through qiime parse mode."""
        ps = _make_ps(with_tax=True)
        p = tmp_path / "qiime_tax.biom"
        pyloseq.write_biom(ps, p, version="2.1")
        ps2 = pyloseq.read_biom(p, parse_taxonomy="qiime")
        assert ps2.tax_table is not None

    def test_v2_values_match_v1(self, tmp_path: Path) -> None:
        ps = _make_ps()
        p1 = tmp_path / "v1.biom"
        p2 = tmp_path / "v2.biom"
        pyloseq.write_biom(ps, p1, version="1.0")
        pyloseq.write_biom(ps, p2, version="2.1")
        ps1 = pyloseq.read_biom(p1)
        ps2 = pyloseq.read_biom(p2)
        s1 = ps1.otu_table.taxa_sums().sort_index()
        s2 = ps2.otu_table.taxa_sums().sort_index()
        np.testing.assert_allclose(s1.values, s2.values, atol=1e-6)


# ===========================================================================
# Ticket 2.3 — QIIME 2 .qza
# ===========================================================================

def _make_feature_table_qza(tmp_path: Path, ps: Phyloseq) -> Path:
    """Create a minimal FeatureTable[Frequency] .qza for testing."""
    import uuid as _uuid_mod

    import yaml

    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "feature-table.qza"

    biom_path = tmp_path / "ft.biom"
    pyloseq.write_biom(ps, biom_path, version="2.1")

    meta = {"uuid": artifact_uuid, "type": "FeatureTable[Frequency]",
            "format": "BIOMV210DirFmt"}
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.write(str(biom_path), f"{artifact_uuid}/data/feature-table.biom")
    return qza_path


def _make_taxonomy_qza(tmp_path: Path, ps: Phyloseq) -> Path:
    import uuid as _uuid_mod

    import yaml

    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "taxonomy.qza"
    if ps.tax_table is None:
        raise ValueError("ps has no tax_table")
    tax_df = ps.tax_table.to_frame()
    taxon_col = tax_df.apply(
        lambda r: "; ".join(f"{k[0].lower()}__{v}" for k, v in r.items() if pd.notna(v)),
        axis=1,
    )
    tsv = "Feature ID\tTaxon\n" + "\n".join(
        f"{idx}\t{taxon}" for idx, taxon in taxon_col.items()
    )
    meta = {"uuid": artifact_uuid, "type": "FeatureData[Taxonomy]",
            "format": "TSVTaxonomyDirectoryFormat"}
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.writestr(f"{artifact_uuid}/data/taxonomy.tsv", tsv)
    return qza_path


def _make_tree_qza(tmp_path: Path, ps: Phyloseq) -> Path:
    import uuid as _uuid_mod

    import yaml

    artifact_uuid = str(_uuid_mod.uuid4())
    qza_path = tmp_path / "tree.qza"
    if ps.phy_tree is None:
        raise ValueError("ps has no phy_tree")
    meta = {"uuid": artifact_uuid, "type": "Phylogeny[Rooted]",
            "format": "NewickDirectoryFormat"}
    with zipfile.ZipFile(str(qza_path), "w") as zf:
        zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        zf.writestr(f"{artifact_uuid}/data/tree.nwk", ps.phy_tree.to_newick())
    return qza_path


class TestQza:
    def test_read_feature_table(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        qza = _make_feature_table_qza(tmp_path, ps)
        ps2 = pyloseq.read_qza(features=qza)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_read_with_taxonomy(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=True)
        ft_qza = _make_feature_table_qza(tmp_path, ps)
        tax_qza = _make_taxonomy_qza(tmp_path, ps)
        ps2 = pyloseq.read_qza(features=ft_qza, taxonomy=tax_qza)
        assert ps2.tax_table is not None
        assert ps2.ntaxa == ps.ntaxa

    def test_read_with_tree(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False, with_tree=True)
        ft_qza = _make_feature_table_qza(tmp_path, ps)
        tree_qza = _make_tree_qza(tmp_path, ps)
        ps2 = pyloseq.read_qza(features=ft_qza, tree=tree_qza)
        assert ps2.phy_tree is not None
        assert ps2.phy_tree.n_tips == ps.ntaxa

    def test_provenance_stashed_in_metadata(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        # Add a dummy provenance file
        import uuid as _uuid_mod

        import yaml
        artifact_uuid = str(_uuid_mod.uuid4())
        qza_path = tmp_path / "prov.qza"
        biom_path = tmp_path / "ft2.biom"
        pyloseq.write_biom(ps, biom_path, version="2.1")
        meta = {"uuid": artifact_uuid, "type": "FeatureTable[Frequency]",
                "format": "BIOMV210DirFmt"}
        with zipfile.ZipFile(str(qza_path), "w") as zf:
            zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
            zf.write(str(biom_path), f"{artifact_uuid}/data/feature-table.biom")
            zf.writestr(f"{artifact_uuid}/provenance/metadata.yaml", yaml.dump({"action": {}}))
        ps2 = pyloseq.read_qza(features=qza_path)
        assert "qza_provenance" in ps2.metadata

    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        qza_out = tmp_path / "out.qza"
        pyloseq.write_qza(ps, qza_out, semantic_type="FeatureTable[Frequency]")
        ps2 = pyloseq.read_qza(features=qza_out)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_unsupported_semantic_type_raises(self, tmp_path: Path) -> None:
        import uuid as _uuid_mod

        import yaml
        artifact_uuid = str(_uuid_mod.uuid4())
        qza_path = tmp_path / "bad.qza"
        meta = {"uuid": artifact_uuid, "type": "SomeUnknown[Type]", "format": "X"}
        with zipfile.ZipFile(str(qza_path), "w") as zf:
            zf.writestr(f"{artifact_uuid}/metadata.yaml", yaml.dump(meta))
        with pytest.raises(ValueError, match="Unsupported"):
            from pyloseq.io._qza import _read_qza_artifact
            _read_qza_artifact(qza_path)

    def test_no_features_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="features="):
            pyloseq.read_qza()


# ===========================================================================
# Ticket 2.4 — QIIME 1
# ===========================================================================

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


class TestQiime1:
    def test_otu_table_only(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        otu_path = tmp_path / "otu_table.txt"
        _write_qiime1_otu_table(otu_path, ps)
        ps2 = pyloseq.read_qiime(otu_path)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_with_taxonomy(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=True)
        otu_path = tmp_path / "otu_tax.txt"
        _write_qiime1_otu_table(otu_path, ps)
        ps2 = pyloseq.read_qiime(otu_path)
        assert ps2.tax_table is not None

    def test_with_mapping(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=True, with_tax=False)
        otu_path = tmp_path / "otu.txt"
        map_path = tmp_path / "mapping.txt"
        _write_qiime1_otu_table(otu_path, ps)
        _write_qiime1_mapping(map_path, ps)
        ps2 = pyloseq.read_qiime(otu_path, mapping=map_path)
        assert ps2.sample_data is not None
        assert ps2.nsamples == ps.nsamples

    def test_abundance_values(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        otu_path = tmp_path / "otu_vals.txt"
        _write_qiime1_otu_table(otu_path, ps)
        ps2 = pyloseq.read_qiime(otu_path)
        orig = ps.otu_table.taxa_sums().sort_index()
        rt = ps2.otu_table.taxa_sums().sort_index()
        np.testing.assert_allclose(orig.values, rt.values, atol=1.0)  # int round-trip

    def test_empty_taxonomy_column_no_taxtable(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        otu_path = tmp_path / "otu_empty_tax.txt"
        df = ps.otu_table.to_dataframe()
        with open(otu_path, "w") as fh:
            fh.write("#OTU ID\t" + "\t".join(df.columns) + "\ttaxonomy\n")
            for otu_id in df.index:
                row = "\t".join(str(int(v)) for v in df.loc[otu_id])
                fh.write(f"{otu_id}\t{row}\t\n")
        ps2 = pyloseq.read_qiime(otu_path)
        assert ps2.tax_table is None


# ===========================================================================
# Ticket 2.5 — mothur
# ===========================================================================

def _write_mothur_shared(path: Path, ps: Phyloseq, cutoff: str = "0.03") -> None:
    df = ps.otu_table.to_dataframe()
    if ps.otu_table.taxa_are_rows:
        df = df.T  # samples as rows
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


class TestMothur:
    def test_shared_only(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        shared = tmp_path / "test.shared"
        _write_mothur_shared(shared, ps)
        ps2 = pyloseq.read_mothur(shared=shared)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_shared_with_taxonomy(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=True)
        shared = tmp_path / "test.shared"
        constax = tmp_path / "test.cons.taxonomy"
        _write_mothur_shared(shared, ps)
        _write_mothur_constaxonomy(constax, ps)
        ps2 = pyloseq.read_mothur(shared=shared, constaxonomy=constax)
        assert ps2.tax_table is not None
        assert ps2.ntaxa == ps.ntaxa

    def test_shared_with_cutoff(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        shared = tmp_path / "multi.shared"
        df = ps.otu_table.to_dataframe()
        if ps.otu_table.taxa_are_rows:
            df = df.T
        otus = list(df.columns)
        with open(shared, "w") as fh:
            fh.write("label\tGroup\tnumOtus\t" + "\t".join(otus) + "\n")
            for cutoff in ("0.03", "0.05"):
                for sample in df.index:
                    vals = "\t".join(str(int(v)) for v in df.loc[sample])
                    fh.write(f"{cutoff}\t{sample}\t{len(otus)}\t{vals}\n")
        ps2 = pyloseq.read_mothur(shared=shared, cutoff="0.05")
        assert ps2.ntaxa == ps.ntaxa

    def test_show_cutoffs(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        shared = tmp_path / "multi2.shared"
        df = ps.otu_table.to_dataframe()
        if ps.otu_table.taxa_are_rows:
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

    def test_no_source_raises(self) -> None:
        with pytest.raises(ValueError):
            pyloseq.read_mothur()


# ===========================================================================
# Ticket 2.7 — CSV/TSV round-trip
# ===========================================================================

class TestCsv:
    def test_otu_only_round_trip(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        pyloseq.to_csv(ps, tmp_path / "out")
        ps2 = pyloseq.read_csv(tmp_path / "out" / "otu_table.tsv")
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples

    def test_full_round_trip(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=True, with_tax=True, with_tree=True)
        out = tmp_path / "full"
        written = pyloseq.to_csv(ps, out)
        ps2 = pyloseq.read_csv(
            written["otu_table"],
            sample_path=written["sample_data"],
            tax_path=written["tax_table"],
            tree_path=written["phy_tree"],
        )
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples
        assert ps2.sample_data is not None
        assert ps2.tax_table is not None
        assert ps2.phy_tree is not None

    def test_abundance_values_preserved(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        out = tmp_path / "vals"
        written = pyloseq.to_csv(ps, out)
        ps2 = pyloseq.read_csv(written["otu_table"])
        orig = ps.otu_table.taxa_sums().sort_index()
        rt = ps2.otu_table.taxa_sums().sort_index()
        np.testing.assert_allclose(orig.values, rt.values, atol=1e-10)

    def test_prefix_option(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        pyloseq.to_csv(ps, tmp_path / "pfx", prefix="myproject_")
        assert (tmp_path / "pfx" / "myproject_otu_table.tsv").exists()

    def test_taxa_are_rows_false(self, tmp_path: Path) -> None:
        ps = _make_ps(with_sam=False, with_tax=False)
        ps.otu_table.taxa_are_rows = False
        out = tmp_path / "flipped"
        written = pyloseq.to_csv(ps, out)
        ps2 = pyloseq.read_csv(written["otu_table"], taxa_are_rows=False)
        assert ps2.ntaxa == ps.ntaxa
        assert ps2.nsamples == ps.nsamples


# ===========================================================================
# Top-level API surface check
# ===========================================================================

def test_io_functions_exported() -> None:
    for name in ["read_biom", "write_biom", "read_qza", "write_qza",
                 "read_qiime", "read_mothur", "show_mothur_cutoffs",
                 "select_mothur_cutoff", "read_csv", "to_csv"]:
        assert hasattr(pyloseq, name), f"pyloseq.{name} not exported"
