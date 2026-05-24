"""BIOM v1 (JSON) and v2 (HDF5) reader and writer.

R reference: phyloseq::import_biom(BIOMfilename, ...)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.sparse as sp

_DEFAULT_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

TaxonomyParser = str | Callable[[Any], dict[str, str]]


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
    ranks = _DEFAULT_RANKS[:n] if n <= len(_DEFAULT_RANKS) else [f"Level{i + 1}" for i in range(n)]
    return dict(zip(ranks, parts, strict=False))


def read_biom(
    path: str | Path,
    parse_taxonomy: TaxonomyParser = "default",
) -> Any:
    """Load a BIOM v1 (JSON) or v2 (HDF5) file into a ``Phyloseq`` object.

    Parameters
    ----------
    path:
        Path to the ``.biom`` file.
    parse_taxonomy:
        How to interpret the ``taxonomy`` field in observation metadata.
        ``"default"`` splits on ``"; "`` or ``";"`` and assigns standard rank
        names.  ``"qiime"`` / ``"greengenes"`` additionally strips rank
        prefixes (``k__``, ``p__``, …).  Pass a callable for custom parsing.

    R reference: phyloseq::import_biom(BIOMfilename, parseFunction=parse_taxonomy)
    """
    import biom

    from pyloseq._otu_table import OtuTable
    from pyloseq._phyloseq import Phyloseq
    from pyloseq._sample_data import SampleData
    from pyloseq._tax_table import TaxTable

    table: biom.Table = biom.load_table(str(path))

    # ---- OTU table -------------------------------------------------------
    # biom.Table.matrix_data is a csc_matrix; to_dataframe gives dense.
    # Preserve sparsity if density < 50 %.
    mat: sp.csc_matrix = table.matrix_data  # type: ignore[assignment]
    taxa_ids = list(table.ids(axis="observation"))
    sample_ids = list(table.ids(axis="sample"))
    nelem = mat.shape[0] * mat.shape[1]
    density = mat.nnz / nelem if nelem > 0 else 1.0

    if density < 0.5:
        otu_data: sp.spmatrix | pd.DataFrame = mat.tocsr()
        otu = OtuTable(otu_data, taxa_are_rows=True)
        otu.taxa_names = pd.Index(taxa_ids)
        otu.sample_names = pd.Index(sample_ids)
    else:
        df = pd.DataFrame(mat.toarray(), index=taxa_ids, columns=sample_ids, dtype=float)
        otu = OtuTable(df, taxa_are_rows=True)

    # ---- Sample metadata -------------------------------------------------
    sam = None
    sam_rows: dict[str, dict[str, Any]] = {}
    for sid in sample_ids:
        meta = table.metadata(sid, axis="sample")
        if meta:
            sam_rows[sid] = dict(meta)
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
    extra: dict[str, Any] = {}
    try:
        import h5py  # type: ignore[import-untyped]

        with h5py.File(str(path), "r") as hf:
            extra = dict(hf.attrs.items())
    except Exception:  # noqa: BLE001
        pass  # v1 JSON or h5py unavailable — skip

    return Phyloseq(otu=otu, sam=sam, tax=tax, metadata=extra if extra else {})


def write_biom(
    ps: Any,
    path: str | Path,
    version: str = "2.1",
) -> None:
    """Write a ``Phyloseq`` object to a BIOM file.

    Parameters
    ----------
    ps:
        The ``Phyloseq`` to serialise.
    path:
        Output file path.
    version:
        ``"2.1"`` (default) writes HDF5; ``"1.0"`` writes JSON.

    R reference: phyloseq::export_biom(x, file)
    """
    import biom
    import h5py  # type: ignore[import-untyped]

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
