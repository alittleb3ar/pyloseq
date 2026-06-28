from pathlib import Path

import pytest
import skbio

from pyloseq import RefSeq


def test_from_fasta_round_trip(tmp_path: Path) -> None:
    seqs = {
        "OTU1": skbio.DNA("ACGT", metadata={"id": "OTU1", "description": ""}),
        "OTU2": skbio.DNA("TTTT", metadata={"id": "OTU2", "description": ""}),
    }
    rs = RefSeq(seqs)
    fasta = tmp_path / "seqs.fasta"
    rs.to_fasta(fasta)
    rs2 = RefSeq.from_fasta(fasta)
    assert set(rs2.taxa_names) == {"OTU1", "OTU2"}
    assert str(rs2["OTU1"]) == "ACGT"
    assert str(rs2["OTU2"]) == "TTTT"


def test_copy_is_independent() -> None:
    rs = RefSeq({"OTU1": skbio.DNA("ACGT")})
    rs2 = rs.copy()
    rs._seqs["OTU2"] = skbio.DNA("TTTT")
    assert "OTU2" not in rs2.taxa_names


def test_len() -> None:
    rs = RefSeq({"OTU1": skbio.DNA("ACGT"), "OTU2": skbio.DNA("TTTT")})
    assert len(rs) == 2


def test_contains() -> None:
    rs = RefSeq({"OTU1": skbio.DNA("ACGT")})
    assert "OTU1" in rs
    assert "OTU2" not in rs


def test_getitem() -> None:
    rs = RefSeq({"OTU1": skbio.DNA("ACGT")})
    assert str(rs["OTU1"]) == "ACGT"


def test_taxa_names_property() -> None:
    rs = RefSeq({"OTU1": skbio.DNA("ACGT"), "OTU2": skbio.DNA("TTTT")})
    assert set(rs.taxa_names) == {"OTU1", "OTU2"}


def test_names_deprecated() -> None:
    rs = RefSeq({"OTU1": skbio.DNA("ACGT")})
    with pytest.warns(DeprecationWarning, match="taxa_names"):
        _ = rs.names


def test_eq_same_sequences() -> None:
    seqs = {"OTU1": skbio.DNA("ACGT")}
    assert RefSeq(seqs) == RefSeq(seqs)


def test_eq_different_sequences() -> None:
    rs1 = RefSeq({"OTU1": skbio.DNA("ACGT")})
    rs2 = RefSeq({"OTU1": skbio.DNA("TTTT")})
    assert rs1 != rs2
