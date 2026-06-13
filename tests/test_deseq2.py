"""Tests for Phyloseq.to_deseq2()"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pyloseq import OtuTable, Phyloseq, SampleData


def test_returns_tuple_of_dataframes(full_ps: Phyloseq) -> None:
    counts, metadata = full_ps.to_deseq2()
    assert isinstance(counts, pd.DataFrame)
    assert isinstance(metadata, pd.DataFrame)


def test_counts_shape_samples_as_rows(full_ps: Phyloseq) -> None:
    counts, _ = full_ps.to_deseq2()
    assert counts.shape == (full_ps.nsamples, full_ps.ntaxa)


def test_counts_index_matches_sample_names(full_ps: Phyloseq) -> None:
    counts, _ = full_ps.to_deseq2()
    assert list(counts.index) == list(full_ps.sample_names)


def test_counts_columns_match_taxa_names(full_ps: Phyloseq) -> None:
    counts, _ = full_ps.to_deseq2()
    assert list(counts.columns) == list(full_ps.taxa_names)


def test_metadata_index_matches_sample_names(full_ps: Phyloseq) -> None:
    _, metadata = full_ps.to_deseq2()
    assert list(metadata.index) == list(full_ps.sample_names)


def test_counts_values_preserved(full_ps: Phyloseq) -> None:
    counts, _ = full_ps.to_deseq2()
    for sample in full_ps.sample_names:
        expected = full_ps.get_sample(sample)
        pd.testing.assert_series_equal(
            counts.loc[sample].rename(None),
            expected.rename(None),
            check_names=False,
        )


def test_taxa_are_rows_false_orientation(ps: Phyloseq) -> None:
    """to_deseq2 must transpose correctly regardless of OTU table orientation."""
    # ps fixture uses taxa_are_rows=True; flip orientation and verify same output
    df = ps.otu_table.to_dataframe().T  # samples × taxa
    otu_samples_as_rows = OtuTable(df, taxa_are_rows=False)
    sam = ps.sample_data
    assert sam is not None
    ps_flipped = Phyloseq(otu=otu_samples_as_rows, sam=sam)

    counts_orig, _ = ps.to_deseq2()
    counts_flipped, _ = ps_flipped.to_deseq2()
    pd.testing.assert_frame_equal(counts_orig, counts_flipped)


def test_raises_without_sample_data(simple_ps: Phyloseq) -> None:
    with pytest.raises(ValueError, match="sample_data is required"):
        simple_ps.to_deseq2()


def test_warns_on_non_integer_counts() -> None:
    df = pd.DataFrame(
        {"S1": [1.5, 2.0], "S2": [3.0, 4.7]},
        index=["OTU1", "OTU2"],
    )
    sam = SampleData(pd.DataFrame({"condition": ["A", "B"]}, index=["S1", "S2"]))
    ps = Phyloseq(otu=OtuTable(df), sam=sam)
    with pytest.warns(UserWarning, match="non-integer"):
        ps.to_deseq2()


def test_no_warning_for_integer_valued_floats(full_ps: Phyloseq) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        full_ps.to_deseq2()
    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
