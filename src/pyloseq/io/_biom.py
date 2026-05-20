"""BIOM v1 (JSON) and v2 (HDF5) reader and writer."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import biom
import h5py
import pandas as pd
import scipy.sparse as sp

from pyloseq._otu_table import _SPARSE_DENSITY_THRESHOLD, OtuTable
from pyloseq._phyloseq import Phyloseq
from pyloseq._sample_data import SampleData
from pyloseq._tax_table import TaxTable

_DEFAULT_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

TaxonomyParser = str | Callable[[Any], dict[str, str]] | None


def _parse_taxonomy_entry(value: Any, mode: TaxonomyParser) -> dict[str, str]:
    if callable(mode):
        return mode(value)
    if isinstance(value, list):
        parts = [str(v).strip() for v in value]
    elif isinstance(value, str):
        sep = "; " if "; " in value else ";"
        parts = [v.strip() for v in value.split(sep)]
    else:
        return {}
    if mode in ("qiime", "greengenes"):
        parts = [re.sub(r"^[a-z]__", "", p).strip() for p in parts]
    n = len(parts)
    ranks = (
        _DEFAULT_RANKS[:n]
        if n <= len(_DEFAULT_RANKS)
        else [f"Level{i + 1}" for i in range(n)]
    )
    return dict(zip(ranks, parts, strict=False))


def _extract_metadata_dict(
    table: Any, ids: list[str], axis: str
) -> dict[str, dict[str, Any]]:
    """Return a {id: metadata_dict} mapping for the given BIOM table axis."""
    rows: dict[str, dict[str, Any]] = {}
    for item_id in ids:
        meta = table.metadata(item_id, axis=axis)
        if meta:
            rows[item_id] = dict(meta)
    return rows


def read_biom(
    path: str | Path,
    parse_taxonomy: TaxonomyParser = "default",
) -> Phyloseq:
    """Load a BIOM v1 (JSON) or v2 (HDF5) file into a ``Phyloseq`` object.

    R reference: phyloseq::import_biom(BIOMfilename, parseFunction=parse_taxonomy)

    Parameters
    ----------
    path:
        Path to the ``.biom`` file.
    parse_taxonomy:
        How to interpret the ``taxonomy`` field in observation metadata.
        ``"default"`` splits on ``"; "`` or ``";"`` and assigns standard rank
        names.  ``"qiime"`` / ``"greengenes"`` additionally strips rank
        prefixes (``k__``, ``p__``, …).  Pass a callable for custom parsing.
    """

    table: biom.Table = biom.load_table(str(path))

    # ---- OTU table -------------------------------------------------------
    # biom.Table.matrix_data is a csc_matrix; to_dataframe gives dense.
    # Preserve sparsity if density < _SPARSE_DENSITY_THRESHOLD.
    mat: sp.csc_matrix = table.matrix_data
    taxa_ids = list(table.ids(axis="observation"))
    sample_ids = list(table.ids(axis="sample"))
    nelem = mat.shape[0] * mat.shape[1]
    density = mat.nnz / nelem if nelem > 0 else 1.0

    if density < _SPARSE_DENSITY_THRESHOLD:
        otu_data: sp.spmatrix | pd.DataFrame = mat.tocsr()
        otu = OtuTable(otu_data, taxa_are_rows=True)
        # Note: these setters enforce uniqueness — a BIOM file with duplicate
        # observation/sample IDs raises here with a clear message.
        otu.taxa_names = pd.Index(taxa_ids)
        otu.sample_names = pd.Index(sample_ids)
    else:
        df = pd.DataFrame(
            mat.toarray(), index=taxa_ids, columns=sample_ids, dtype=float
        )
        otu = OtuTable(df, taxa_are_rows=True)

    # ---- Sample metadata -------------------------------------------------
    sam = None
    sam_rows = _extract_metadata_dict(table, sample_ids, axis="sample")
    if sam_rows:
        sam = SampleData(pd.DataFrame.from_dict(sam_rows, orient="index"))

    # ---- Tax table -------------------------------------------------------
    tax = None
    if parse_taxonomy is not None:
        tax_rows: dict[str, dict[str, str]] = {}
        for oid in taxa_ids:
            meta = table.metadata(oid, axis="observation")
            if meta and "taxonomy" in meta:
                tax_rows[oid] = _parse_taxonomy_entry(meta["taxonomy"], parse_taxonomy)
        if tax_rows:
            tax = TaxTable(pd.DataFrame.from_dict(tax_rows, orient="index"))

    # ---- HDF5 attributes (v2 only) ---------------------------------------
    # h5py is imported at module load, so only OSError is reachable here
    # (e.g. a v1 JSON file, which is not valid HDF5).
    extra: dict[str, Any] = {}
    try:
        with h5py.File(str(path), "r") as hf:
            extra = dict(hf.attrs.items())
    except OSError:
        pass  # not a valid HDF5 file (e.g. BIOM v1 JSON)

    return Phyloseq(otu=otu, sam=sam, tax=tax, metadata=extra if extra else {})


def write_biom(
    ps: Phyloseq,
    path: str | Path,
    version: str = "2.1",
) -> None:
    """Write a ``Phyloseq`` object to a BIOM file.

    R reference: phyloseq::export_biom(x, file)

    Parameters
    ----------
    ps:
        The ``Phyloseq`` to serialise.
    path:
        Output file path.
    version:
        ``"2.1"`` (default) writes HDF5; ``"1.0"`` writes JSON.
    """

    df = ps.otu_table.to_dataframe()
    if not ps.otu_table.taxa_are_rows:
        df = df.T

    # Observation metadata (taxonomy)
    obs_meta: list[dict[str, Any]] | None = None
    if ps.tax_table is not None:
        tax_df = ps.tax_table.to_frame()
        obs_meta = []
        for oid in df.index:
            if oid in tax_df.index:
                obs_meta.append({"taxonomy": list(tax_df.loc[oid].fillna(""))})
            else:
                obs_meta.append({})

    # Sample metadata
    sam_meta: list[dict[str, Any]] | None = None
    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        sam_meta = []
        for sid in df.columns:
            if sid in sam_df.index:
                sam_meta.append(dict(sam_df.loc[sid]))
            else:
                sam_meta.append({})

    mat = sp.csc_matrix(df.values.astype(float))
    table = biom.Table(
        mat,
        observation_ids=list(df.index),
        sample_ids=list(df.columns),
        observation_metadata=obs_meta,
        sample_metadata=sam_meta,
    )

    path = Path(path)
    if version.startswith("1"):
        with open(path, "w") as fh:
            table.to_json("pyloseq", direct_io=fh)
    else:
        with h5py.File(str(path), "w") as hf:
            table.to_hdf5(hf, "pyloseq")
