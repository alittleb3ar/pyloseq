"""Tests for multi_tax_test (differential abundance, multiple hypothesis testing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyloseq
from pyloseq import OtuTable, Phyloseq, SampleData, TaxTable, multi_tax_test
from pyloseq._hypothesis import _holm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ps(
    signal_taxa: int = 2,
    noise_taxa: int = 8,
    n_per_group: int = 5,
    rng_seed: int = 0,
) -> Phyloseq:
    """Phyloseq with a known differential signal between group A and group B."""
    rng = np.random.default_rng(rng_seed)
    n = n_per_group * 2
    M = signal_taxa + noise_taxa

    taxa = [f"OTU{i}" for i in range(M)]
    samples_a = [f"A{i}" for i in range(n_per_group)]
    samples_b = [f"B{i}" for i in range(n_per_group)]
    samples = samples_a + samples_b

    counts = rng.integers(1, 50, size=(M, n)).astype(float)
    # Inject strong signal into the first `signal_taxa` taxa
    counts[:signal_taxa, :n_per_group] += 200
    counts[:signal_taxa, n_per_group:] = rng.integers(
        1, 5, size=(signal_taxa, n_per_group)
    )

    otu_df = pd.DataFrame(counts, index=taxa, columns=samples)
    sam_df = pd.DataFrame(
        {"Group": ["A"] * n_per_group + ["B"] * n_per_group},
        index=samples,
    )
    tax_df = pd.DataFrame(
        {"Phylum": ["Firm"] * (M // 2) + ["Bact"] * (M - M // 2)},
        index=taxa,
    )
    return Phyloseq(
        otu=OtuTable(otu_df, taxa_are_rows=True),
        sam=SampleData(sam_df),
        tax=TaxTable(tax_df),
    )


@pytest.fixture
def ps_two_groups() -> Phyloseq:
    return _make_ps()


# ---------------------------------------------------------------------------
# Basic output shape and types
# ---------------------------------------------------------------------------


def test_returns_dataframe(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert isinstance(result, pd.DataFrame)


def test_row_count_equals_ntaxa(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert len(result) == ps_two_groups.ntaxa


def test_required_columns_present(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert {"statistic", "rawp", "adjp", "mean_A", "mean_B"}.issubset(result.columns)


def test_sorted_by_adjp(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert list(result["adjp"]) == sorted(result["adjp"])


def test_pvalues_in_unit_interval(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert (result["rawp"].between(0, 1)).all()
    assert (result["adjp"].between(0, 1)).all()


def test_adjp_ge_rawp_for_bonferroni(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group", method="bonferroni")
    # Bonferroni adjusted p >= raw p (inflates, never deflates)
    assert (result["adjp"] >= result["rawp"] - 1e-12).all()


def test_mean_columns_are_nonnegative(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert (result["mean_A"] >= 0).all()
    assert (result["mean_B"] >= 0).all()


# ---------------------------------------------------------------------------
# Signal detection: strongly separated taxa should rank at the top
# ---------------------------------------------------------------------------


def test_signal_taxa_rank_first(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    top_two = set(result.head(2).index)
    assert top_two == {"OTU0", "OTU1"}, f"Expected signal taxa first, got {top_two}"


def test_signal_taxa_have_low_adjp(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(ps_two_groups, "Group")
    assert result.loc["OTU0", "adjp"] < 0.05
    assert result.loc["OTU1", "adjp"] < 0.05


# ---------------------------------------------------------------------------
# All correction methods run without error and return valid p-values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["BH", "BY", "holm", "bonferroni", "westfall_young"])
def test_all_methods_valid(ps_two_groups: Phyloseq, method: str) -> None:
    result = multi_tax_test(ps_two_groups, "Group", method=method, n_permutations=50)
    assert (result["adjp"].between(0, 1)).all()
    assert not result["adjp"].isna().any()


@pytest.mark.parametrize("test", ["t", "wilcoxon"])
def test_all_tests_valid(ps_two_groups: Phyloseq, test: str) -> None:
    result = multi_tax_test(ps_two_groups, "Group", test=test)
    assert not result["adjp"].isna().any()


# ---------------------------------------------------------------------------
# Westfall-Young: reproducibility and monotonicity
# ---------------------------------------------------------------------------


def test_westfall_young_reproducible(ps_two_groups: Phyloseq) -> None:
    r1 = multi_tax_test(
        ps_two_groups, "Group", method="westfall_young", n_permutations=200, rng_seed=7
    )
    r2 = multi_tax_test(
        ps_two_groups, "Group", method="westfall_young", n_permutations=200, rng_seed=7
    )
    pd.testing.assert_frame_equal(r1, r2)


def test_westfall_young_adjp_monotone(ps_two_groups: Phyloseq) -> None:
    result = multi_tax_test(
        ps_two_groups, "Group", method="westfall_young", n_permutations=200
    )
    # Already sorted by adjp; values must be non-decreasing
    adjp = result["adjp"].values
    assert np.all(
        np.diff(adjp) >= -1e-12
    ), "Westfall-Young adjusted p-values are not monotone"


# ---------------------------------------------------------------------------
# Holm helper: known example
# ---------------------------------------------------------------------------


def test_holm_known_example() -> None:
    # Example from Holm (1979): 4 raw p-values
    pvals = np.array([0.01, 0.04, 0.03, 0.005])
    adjusted = _holm(pvals)
    # Sorted: 0.005, 0.01, 0.03, 0.04
    # Factors: ×4, ×3, ×2, ×1 → 0.02, 0.03, 0.06, 0.04
    # Monotone: 0.02, 0.03, 0.06, 0.06
    # p[0.005] → 0.02, p[0.01] → 0.03, p[0.03] → 0.06, p[0.04] → 0.06
    expected = np.array([0.03, 0.06, 0.06, 0.02])  # in original order
    np.testing.assert_allclose(adjusted, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# BH: adjusted p-values should be >= raw (BH is anti-conservative relative to Bonferroni)
# ---------------------------------------------------------------------------


def test_bh_le_bonferroni(ps_two_groups: Phyloseq) -> None:
    bh = multi_tax_test(ps_two_groups, "Group", method="BH")
    bon = multi_tax_test(ps_two_groups, "Group", method="bonferroni")
    # BH controls FDR (weaker), so its adjusted p-values should be <= Bonferroni's
    assert (bh["adjp"].values <= bon.reindex(bh.index)["adjp"].values + 1e-12).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_sample_data_raises() -> None:
    otu_df = pd.DataFrame({"S1": [1.0], "S2": [2.0]}, index=["OTU1"])
    ps = Phyloseq(otu=OtuTable(otu_df, taxa_are_rows=True))
    with pytest.raises(pyloseq.pyloseqValidationError, match="sample_data"):
        multi_tax_test(ps, "Group")


def test_unknown_grouping_var_raises(ps_two_groups: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError, match="not found"):
        multi_tax_test(ps_two_groups, "DoesNotExist")


def test_more_than_two_groups_raises() -> None:
    otu_df = pd.DataFrame({"S1": [1.0], "S2": [2.0], "S3": [3.0]}, index=["OTU1"])
    sam_df = pd.DataFrame({"Group": ["A", "B", "C"]}, index=["S1", "S2", "S3"])
    ps = Phyloseq(otu=OtuTable(otu_df, taxa_are_rows=True), sam=SampleData(sam_df))
    with pytest.raises(pyloseq.pyloseqValidationError, match="2 non-NaN groups"):
        multi_tax_test(ps, "Group")


def test_single_sample_per_group_raises() -> None:
    otu_df = pd.DataFrame({"S1": [1.0], "S2": [2.0]}, index=["OTU1"])
    sam_df = pd.DataFrame({"Group": ["A", "B"]}, index=["S1", "S2"])
    ps = Phyloseq(otu=OtuTable(otu_df, taxa_are_rows=True), sam=SampleData(sam_df))
    with pytest.raises(pyloseq.pyloseqValidationError, match="at least 2 samples"):
        multi_tax_test(ps, "Group")


def test_unknown_test_raises(ps_two_groups: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError, match="Unknown test"):
        multi_tax_test(ps_two_groups, "Group", test="bad")  # type: ignore[arg-type]


def test_unknown_method_raises(ps_two_groups: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError, match="Unknown method"):
        multi_tax_test(ps_two_groups, "Group", method="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Samples with NaN grouping are silently dropped
# ---------------------------------------------------------------------------


def test_nan_samples_dropped() -> None:
    otu_df = pd.DataFrame(
        {"S1": [10.0], "S2": [1.0], "S3": [200.0], "S4": [2.0], "S5": [5.0]},
        index=["OTU1"],
    )
    sam_df = pd.DataFrame(
        {"Group": ["A", "A", "B", "B", None]},
        index=["S1", "S2", "S3", "S4", "S5"],
    )
    ps = Phyloseq(otu=OtuTable(otu_df, taxa_are_rows=True), sam=SampleData(sam_df))
    result = multi_tax_test(ps, "Group")
    assert len(result) == 1
    assert not result["adjp"].isna().any()


# ---------------------------------------------------------------------------
# Constant (all-zero) taxon: NaN replaced with 1.0
# ---------------------------------------------------------------------------


def test_constant_taxon_handled() -> None:
    otu_df = pd.DataFrame(
        {
            "S1": [100.0, 0.0],
            "S2": [110.0, 0.0],
            "S3": [5.0, 0.0],
            "S4": [3.0, 0.0],
        },
        index=["OTU_signal", "OTU_zero"],
    )
    sam_df = pd.DataFrame(
        {"Group": ["A", "A", "B", "B"]},
        index=["S1", "S2", "S3", "S4"],
    )
    ps = Phyloseq(otu=OtuTable(otu_df, taxa_are_rows=True), sam=SampleData(sam_df))
    result = multi_tax_test(ps, "Group")
    assert result.loc["OTU_zero", "rawp"] == 1.0
    assert not result["adjp"].isna().any()
