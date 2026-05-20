"""Verifies all I/O functions are exported from the top-level pyloseq namespace."""

from __future__ import annotations

import pyloseq


def test_io_functions_exported() -> None:
    for name in [
        "read_biom",
        "write_biom",
        "read_qiime",
        "read_mothur",
        "show_mothur_cutoffs",
        "select_mothur_cutoff",
        "read_csv",
        "to_csv",
    ]:
        assert hasattr(pyloseq, name), f"pyloseq.{name} not exported"
