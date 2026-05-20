import pandas as pd
import pytest

from pyloseq import SampleData
from pyloseq.datasets import load_enterotype_reference


def test_basic_construction() -> None:
    df = pd.DataFrame({"age": [25, 30], "site": ["gut", "oral"]}, index=["S1", "S2"])
    sd = SampleData(df)
    assert len(sd) == 2
    assert list(sd.variables) == ["age", "site"]


def test_names() -> None:
    df = pd.DataFrame({"x": [1]}, index=["MySample"])
    sd = SampleData(df)
    assert list(sd.names) == ["MySample"]


def test_to_frame_is_copy() -> None:
    df = pd.DataFrame({"x": [1]}, index=["S1"])
    sd = SampleData(df)
    frame = sd.to_frame()
    frame["x"] = 999
    assert sd.to_frame()["x"].iloc[0] == 1


def test_categorical_dtype_preserved() -> None:
    df = pd.DataFrame(
        {"grp": pd.Categorical(["a", "b", "a"])}, index=["S1", "S2", "S3"]
    )
    sd = SampleData(df)
    assert isinstance(sd.to_frame()["grp"].dtype, pd.CategoricalDtype)


def test_enterotype_shape() -> None:
    ref = load_enterotype_reference()
    sd = SampleData(ref["sample_data"])
    assert sd.to_frame().shape == (280, 9)


def test_sample_names_property() -> None:
    df = pd.DataFrame({"x": [1, 2]}, index=["S1", "S2"])
    sd = SampleData(df)
    assert list(sd.sample_names) == ["S1", "S2"]


def test_duplicate_sample_names_raises() -> None:
    df = pd.DataFrame({"x": [1, 2]}, index=["S1", "S1"])
    with pytest.raises(ValueError):
        SampleData(df)


def test_repr() -> None:
    sd = SampleData(pd.DataFrame({"a": [1], "b": [2]}, index=["S1"]))
    r = repr(sd)
    assert "1" in r and "samples" in r
    assert "2" in r and "variables" in r


def test_eq_same_data() -> None:
    df = pd.DataFrame({"x": [1, 2]}, index=["S1", "S2"])
    assert SampleData(df) == SampleData(df)


def test_eq_different_data() -> None:
    df1 = pd.DataFrame({"x": [1]}, index=["S1"])
    df2 = pd.DataFrame({"x": [9]}, index=["S1"])
    assert SampleData(df1) != SampleData(df2)


def test_non_dataframe_raises() -> None:
    with pytest.raises(TypeError):
        SampleData({"x": [1]})
