import pandas as pd
import pytest

from pyloseq import TaxTable
from pyloseq.datasets import load_global_patterns_reference


def test_basic_construction() -> None:
    df = pd.DataFrame({"Phylum": ["Firmicutes"], "Genus": ["Bacillus"]}, index=["OTU1"])
    tt = TaxTable(df)
    assert tt.rank_names == ["Phylum", "Genus"]
    assert list(tt.taxa_names) == ["OTU1"]


def test_to_frame_is_copy() -> None:
    df = pd.DataFrame({"Phylum": ["Firmicutes"]}, index=["OTU1"])
    tt = TaxTable(df)
    frame = tt.to_frame()
    frame["Phylum"] = "Other"
    assert tt.to_frame()["Phylum"].iloc[0] == "Firmicutes"


def test_global_patterns_shape() -> None:
    ref = load_global_patterns_reference()
    tt = TaxTable(ref["tax_table"])
    assert tt.to_frame().shape == (19216, 7)


def test_len() -> None:
    df = pd.DataFrame({"Phylum": ["A", "B"]}, index=["OTU1", "OTU2"])
    assert len(TaxTable(df)) == 2


def test_repr() -> None:
    df = pd.DataFrame({"Phylum": ["A"]}, index=["OTU1"])
    r = repr(TaxTable(df))
    assert "Phylum" in r


def test_eq_same_data() -> None:
    df = pd.DataFrame({"Phylum": ["A"]}, index=["OTU1"])
    assert TaxTable(df) == TaxTable(df)


def test_eq_different_data() -> None:
    df1 = pd.DataFrame({"Phylum": ["A"]}, index=["OTU1"])
    df2 = pd.DataFrame({"Phylum": ["B"]}, index=["OTU1"])
    assert TaxTable(df1) != TaxTable(df2)


def test_duplicate_taxa_raises() -> None:
    df = pd.DataFrame({"Phylum": ["A", "B"]}, index=["OTU1", "OTU1"])
    with pytest.raises(ValueError):
        TaxTable(df)


def test_global_patterns_specific_value() -> None:
    ref = load_global_patterns_reference()
    tt = TaxTable(ref["tax_table"])
    assert tt.to_frame().loc["549322", "Kingdom"] == "Archaea"
    assert tt.to_frame().loc["549322", "Phylum"] == "Crenarchaeota"
