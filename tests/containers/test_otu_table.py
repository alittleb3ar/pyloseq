import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from pyloseq import OtuTable
from pyloseq.datasets import load_global_patterns_reference


def test_construct_from_dataframe() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], index=["OTU1", "OTU2"], columns=["S1", "S2"])
    ot = OtuTable(df, taxa_are_rows=True)
    assert ot.ntaxa == 2
    assert ot.nsamples == 2
    assert list(ot.taxa_names) == ["OTU1", "OTU2"]
    assert list(ot.sample_names) == ["S1", "S2"]


def test_construct_from_ndarray() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    ot = OtuTable(arr, taxa_are_rows=True)
    assert ot.ntaxa == 2
    assert ot.nsamples == 2


def test_construct_from_list() -> None:
    ot = OtuTable([[1, 2], [3, 4]])
    assert ot.ntaxa == 2


def test_construct_from_sparse_produces_correct_values() -> None:
    mat = sp.csr_matrix(np.array([[1, 0], [0, 4]]))
    ot = OtuTable(mat, taxa_are_rows=True)
    assert ot.ntaxa == 2
    df = ot.to_dataframe()
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 4


def test_orientation_flip_preserves_dataframe_values() -> None:
    df = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        index=["OTU1", "OTU2"],
        columns=["S1", "S2", "S3"],
    )
    ot = OtuTable(df, taxa_are_rows=True)
    original_df = ot.to_dataframe().copy()

    ot.taxa_are_rows = False
    ot.taxa_are_rows = True  # flip back

    pd.testing.assert_frame_equal(ot.to_dataframe(), original_df)


def test_orientation_flip_preserves_logical_counts() -> None:
    df = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
    ot = OtuTable(df, taxa_are_rows=True)
    assert ot.ntaxa == 1
    assert ot.nsamples == 2
    assert list(ot.taxa_names) == ["OTU1"]
    assert list(ot.sample_names) == ["S1", "S2"]

    ot.taxa_are_rows = False
    assert ot.ntaxa == 1
    assert ot.nsamples == 2
    assert list(ot.taxa_names) == ["OTU1"]
    assert list(ot.sample_names) == ["S1", "S2"]


def test_taxa_sums() -> None:
    df = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["OTU1", "OTU2"],
        columns=["S1", "S2"],
    )
    ot = OtuTable(df, taxa_are_rows=True)
    sums = ot.taxa_sums()
    assert sums["OTU1"] == 30
    assert sums["OTU2"] == 70


def test_sample_sums() -> None:
    df = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["OTU1", "OTU2"],
        columns=["S1", "S2"],
    )
    ot = OtuTable(df, taxa_are_rows=True)
    sums = ot.sample_sums()
    assert sums["S1"] == 40
    assert sums["S2"] == 60


def test_sums_consistent_after_flip() -> None:
    df = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["OTU1", "OTU2"],
        columns=["S1", "S2"],
    )
    ot = OtuTable(df, taxa_are_rows=True)
    taxa_sums_before = ot.taxa_sums()
    sample_sums_before = ot.sample_sums()

    ot.taxa_are_rows = False

    pd.testing.assert_series_equal(ot.taxa_sums(), taxa_sums_before, check_names=False)
    pd.testing.assert_series_equal(
        ot.sample_sums(), sample_sums_before, check_names=False
    )


def test_sparse_large_low_density() -> None:

    rng = np.random.default_rng(0)
    n_taxa, n_samples = 100_000, 1_000
    mat = sp.random(n_taxa, n_samples, density=0.01, format="csr", random_state=rng)
    tracemalloc.start()
    ot = OtuTable(mat, taxa_are_rows=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert ot._is_sparse
    assert peak < 50 * 1024 * 1024, f"Peak memory {peak / 1e6:.1f} MB exceeded 50 MB"


@given(
    np_arrays(
        np.float64,
        st.tuples(st.integers(1, 10), st.integers(1, 10)),
        elements=st.floats(0, 1e6, allow_nan=False, allow_infinity=False),
    ),
    st.booleans(),
)
@settings(max_examples=50)
def test_nsamples_ntaxa_product(arr: np.ndarray, taxa_are_rows: bool) -> None:
    ot = OtuTable(arr, taxa_are_rows=taxa_are_rows)
    assert ot.nsamples * ot.ntaxa == arr.size


def test_taxa_sums_match_r_reference() -> None:

    ref = load_global_patterns_reference()
    ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
    python_sums = ot.taxa_sums().sort_index()
    r_sums = ref["taxa_sums"].sort_index()
    np.testing.assert_allclose(
        python_sums.values, r_sums.values, atol=1e-10, rtol=1e-12
    )


def test_sample_sums_match_r_reference() -> None:
    ref = load_global_patterns_reference()
    ot = OtuTable(ref["otu_table"], taxa_are_rows=True)
    python_sums = ot.sample_sums().sort_index()
    r_sums = ref["sample_sums"].sort_index()
    np.testing.assert_allclose(
        python_sums.values, r_sums.values, atol=1e-10, rtol=1e-12
    )


def test_copy_is_independent() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], index=["OTU1", "OTU2"], columns=["S1", "S2"])
    ot = OtuTable(df)
    ot2 = ot.copy()
    ot2.taxa_names = pd.Index(["X", "Y"])
    assert list(ot.taxa_names) == ["OTU1", "OTU2"]


def test_eq_same_data() -> None:
    df = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
    assert OtuTable(df) == OtuTable(df)


def test_eq_different_values() -> None:
    df1 = pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"])
    df2 = pd.DataFrame([[9, 2]], index=["OTU1"], columns=["S1", "S2"])
    assert OtuTable(df1) != OtuTable(df2)


def test_repr_contains_dimensions() -> None:
    ot = OtuTable(
        pd.DataFrame([[1, 2], [3, 4]], index=["OTU1", "OTU2"], columns=["S1", "S2"])
    )
    r = repr(ot)
    assert "2" in r
    assert "taxa" in r.lower() or "×" in r


def test_taxa_names_setter_wrong_length_raises() -> None:
    ot = OtuTable(pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"]))
    with pytest.raises(ValueError):
        ot.taxa_names = pd.Index(["X", "Y"])


def test_sample_names_setter_wrong_length_raises() -> None:
    ot = OtuTable(pd.DataFrame([[1, 2]], index=["OTU1"], columns=["S1", "S2"]))
    with pytest.raises(ValueError):
        ot.sample_names = pd.Index(["S1"])


def test_duplicate_taxa_in_dataframe_raises() -> None:
    df = pd.DataFrame([[1, 2], [3, 4]], index=["OTU1", "OTU1"], columns=["S1", "S2"])
    with pytest.raises(ValueError):
        OtuTable(df)
