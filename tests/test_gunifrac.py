"""Tests for gunifrac — Generalized UniFrac distance family (Chen et al. 2012)."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest
import skbio.tree
from skbio.stats.distance import DistanceMatrix

import pyloseq
from pyloseq import GUnifracResult, OtuTable, Phyloseq, PhyTree, gunifrac

# ---------------------------------------------------------------------------
# Fixtures and expected values
# ---------------------------------------------------------------------------

# Simple 3-tip tree for two-sample hand-verifiable calculations.
# Branches (postorder, skipping the parentless root):
#   A   (length 0.1)   S1 cumulative prop = 0.25,  S2 = 0.00
#   B   (length 0.2)   S1 = 0.00,                  S2 = 0.40
#   AB  (length 0.3)   S1 = 0.25,                  S2 = 0.40
#   C   (length 0.4)   S1 = 0.75,                  S2 = 0.60
_SIMPLE_NEWICK = "((A:0.1,B:0.2):0.3,C:0.4);"


def _make_simple_ps(counts: dict[str, list[int]]) -> Phyloseq:
    df = pd.DataFrame(counts, index=["A", "B", "C"])
    tree_node = skbio.tree.TreeNode.read(
        StringIO(_SIMPLE_NEWICK), format="newick", convert_underscores=False
    )
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        tree=PhyTree(tree_node),
    )


# S1: A=10, B=0, C=30  →  proportions 0.25, 0.00, 0.75
# S2: A=0,  B=20, C=30 →  proportions 0.00, 0.40, 0.60
#
# Hand-computed d(S1, S2):
#   d_1   = (0.1·0.25 + 0.2·0.4 + 0.3·0.15 + 0.4·0.15)
#           / (0.1·0.25 + 0.2·0.4 + 0.3·0.65 + 0.4·1.35)
#         = 0.21 / 0.84 = 0.25
#   d_0.5 formula: num = sum(l·|diff|/sqrt(s)), den = sum(l·sqrt(s))
#         num = 0.1·sqrt(0.25) + 0.2·sqrt(0.4) + 0.3·0.15/sqrt(0.65) + 0.4·0.15/sqrt(1.35)
#         den = 0.1·sqrt(0.25) + 0.2·sqrt(0.4) + 0.3·sqrt(0.65) + 0.4·sqrt(1.35)
#   d_0   = sum(l·|diff|/s) / sum(l) (relative-difference weighted)
#         = (0.1·1 + 0.2·1 + 0.3·0.15/0.65 + 0.4·0.15/1.35) / 1.0
#   d_UW  = binarized: only A (S1 only) and B (S2 only) are unique branches
#         = (0.1 + 0.2) / (0.1 + 0.2 + 0.3 + 0.4) = 0.3
#   d_VAW = sqrt(0.12205) / sqrt(0.84) ≈ 0.38118
_SIMPLE_2S = {"S1": [10, 0, 30], "S2": [0, 20, 30]}

_EXPECT_D1 = 0.25
_EXPECT_D05 = (
    0.1 * np.sqrt(0.25)
    + 0.2 * np.sqrt(0.4)
    + 0.045 / np.sqrt(0.65)
    + 0.06 / np.sqrt(1.35)
) / (
    0.1 * np.sqrt(0.25) + 0.2 * np.sqrt(0.4) + 0.3 * np.sqrt(0.65) + 0.4 * np.sqrt(1.35)
)
# d_0: sum(l · |diff| / s) / sum(l) over branches where s > 0
_EXPECT_D0 = (
    0.1 * 1.0 + 0.2 * 1.0 + 0.3 * (0.15 / 0.65) + 0.4 * (0.15 / 1.35)
)  # den = 1.0
# d_UW: only branches unique to one sample (A: S1 only, B: S2 only)
_EXPECT_DUW = 0.3  # (0.1 + 0.2) / (0.1 + 0.2 + 0.3 + 0.4)
_EXPECT_DVAW = np.sqrt(0.122051) / np.sqrt(0.84)


# Chen et al. (2012) Table 1 example: 4 tips, unit branch lengths, 3 samples.
# SA: OTU1=1, OTU2=1, OTU3=0, OTU4=0  →  props 0.5, 0.5, 0, 0
# SB: OTU1=1, OTU2=0, OTU3=1, OTU4=0  →  props 0.5, 0, 0.5, 0
# SC: OTU1=0, OTU2=0, OTU3=1, OTU4=1  →  props 0, 0, 0.5, 0.5
#
# Branches (postorder: OTU1, OTU2, nodeAB, OTU3, OTU4, nodeCD):
#   cumulative props SA: [0.5, 0.5, 1.0, 0.0, 0.0, 0.0]
#   cumulative props SB: [0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
#   cumulative props SC: [0.0, 0.0, 0.0, 0.5, 0.5, 1.0]
#
# Expected d(SA, SB): d_1=0.5, d_0=2/3, d_UW=3/5=0.6, d_VAW=sqrt(5/3)/2≈0.6455
# Expected d(SA, SC): d_1=1.0, d_0=1.0, d_UW=1.0, d_VAW=1.0  (completely disjoint)
# Expected d(SB, SC): d_1=0.5, d_0=2/3, d_UW=3/5=0.6, d_VAW=sqrt(5/3)/2≈0.6455
#
# d_0 derivation SA-SB (sum over 5 branches where pA+pB>0; OTU4 masked):
#   OTU1: |0.5-0.5|/1.0=0, OTU2: |0.5-0|/0.5=1, nodeAB: |1.0-0.5|/1.5=1/3
#   OTU3: |0-0.5|/0.5=1, nodeCD: |0-0.5|/0.5=1  →  (0+1+1/3+1+1)/5 = 10/15 = 2/3
# d_UW derivation SA-SB (binarized):
#   unique branches: OTU2 (SA only), OTU3 (SB only), nodeCD (SB only)  →  3/5
_CHEN_NEWICK = "((OTU1:1,OTU2:1):1,(OTU3:1,OTU4:1):1);"
_CHEN_COUNTS: dict[str, list[int]] = {
    "SA": [1, 1, 0, 0],
    "SB": [1, 0, 1, 0],
    "SC": [0, 0, 1, 1],
}
_CHEN_TAXA = ["OTU1", "OTU2", "OTU3", "OTU4"]


def _make_chen_ps() -> Phyloseq:
    df = pd.DataFrame(_CHEN_COUNTS, index=_CHEN_TAXA)
    tree_node = skbio.tree.TreeNode.read(
        StringIO(_CHEN_NEWICK), format="newick", convert_underscores=False
    )
    return Phyloseq(
        otu=OtuTable(df, taxa_are_rows=True),
        tree=PhyTree(tree_node),
    )


# ===========================================================================
# Return type and key structure
# ===========================================================================


def test_gunifrac_returns_gunifrac_result() -> None:
    ps = _make_simple_ps(_SIMPLE_2S)
    result = gunifrac(ps)
    assert isinstance(result, GUnifracResult)
    for v in result.values():
        assert isinstance(v, DistanceMatrix)


def test_gunifrac_default_keys() -> None:
    ps = _make_simple_ps(_SIMPLE_2S)
    assert set(gunifrac(ps).keys()) == {"d_0", "d_0.5", "d_1", "d_UW", "d_VAW"}


def test_gunifrac_custom_alpha_keys() -> None:
    ps = _make_simple_ps(_SIMPLE_2S)
    result = gunifrac(ps, alpha=(0.25, 0.75))
    assert "d_0.25" in result
    assert "d_0.75" in result
    assert "d_UW" in result
    assert "d_VAW" in result
    # Default alpha keys must NOT appear when not requested
    assert "d_0" not in result
    assert "d_0.5" not in result
    assert "d_1" not in result


def test_gunifrac_single_alpha_key() -> None:
    ps = _make_simple_ps(_SIMPLE_2S)
    result = gunifrac(ps, alpha=(1,))
    assert set(result.keys()) == {"d_1", "d_UW", "d_VAW"}


# ===========================================================================
# Distance matrix properties
# ===========================================================================


@pytest.mark.parametrize("key", ["d_0", "d_0.5", "d_1", "d_UW", "d_VAW"])
def test_gunifrac_dm_shape(ps_with_tree: Phyloseq, key: str) -> None:
    dm = gunifrac(ps_with_tree)[key]
    assert dm.shape == (ps_with_tree.nsamples, ps_with_tree.nsamples)


@pytest.mark.parametrize("key", ["d_0", "d_0.5", "d_1", "d_UW", "d_VAW"])
def test_gunifrac_dm_zero_diagonal(ps_with_tree: Phyloseq, key: str) -> None:
    arr = np.array(gunifrac(ps_with_tree)[key].data)
    np.testing.assert_allclose(np.diag(arr), 0.0, atol=1e-12)


@pytest.mark.parametrize("key", ["d_0", "d_0.5", "d_1", "d_UW", "d_VAW"])
def test_gunifrac_dm_symmetric(ps_with_tree: Phyloseq, key: str) -> None:
    arr = np.array(gunifrac(ps_with_tree)[key].data)
    np.testing.assert_allclose(arr, arr.T, atol=1e-12)


@pytest.mark.parametrize("key", ["d_0", "d_0.5", "d_1", "d_UW", "d_VAW"])
def test_gunifrac_values_in_unit_range(ps_with_tree: Phyloseq, key: str) -> None:
    arr = np.array(gunifrac(ps_with_tree)[key].data)
    assert arr.min() >= -1e-12
    assert arr.max() <= 1.0 + 1e-12


def test_gunifrac_sample_ids_match(ps_with_tree: Phyloseq) -> None:
    expected = set(ps_with_tree.sample_names)
    for dm in gunifrac(ps_with_tree).values():
        assert set(dm.ids) == expected


# ===========================================================================
# Error handling
# ===========================================================================


def test_gunifrac_requires_tree(ps: Phyloseq) -> None:
    with pytest.raises(pyloseq.pyloseqValidationError):
        gunifrac(ps)


def test_gunifrac_no_matching_tips_raises() -> None:
    df = pd.DataFrame({"S1": [5, 3], "S2": [2, 6]}, index=["X1", "X2"])
    tree_node = skbio.tree.TreeNode.read(
        StringIO("(A:0.1,B:0.2);"), format="newick", convert_underscores=False
    )
    ps = Phyloseq(otu=OtuTable(df, taxa_are_rows=True), tree=PhyTree(tree_node))
    with pytest.raises(pyloseq.pyloseqValidationError):
        gunifrac(ps)


# ===========================================================================
# d_0 and d_UW are distinct metrics (d_0 uses actual proportions, d_UW binarizes)
# ===========================================================================


def test_gunifrac_d0_differs_from_d_uw() -> None:
    """d_0 and d_UW are distinct: d_0 adds fractional credit for shared branches
    (where proportions differ), so d_0 >= d_UW when overlap exists."""
    ps = _make_simple_ps(_SIMPLE_2S)
    result = gunifrac(ps)
    # S1 and S2 share OTU C → AB and C are shared branches counted in d_0 but not d_UW
    assert result["d_0"]["S1", "S2"] > result["d_UW"]["S1", "S2"]


# ===========================================================================
# Identical samples → zero distance
# ===========================================================================


def test_gunifrac_identical_samples_zero() -> None:
    ps = _make_simple_ps({"S1": [10, 20, 30], "S2": [10, 20, 30]})
    for key, dm in gunifrac(ps, alpha=(0, 0.5, 1)).items():
        np.testing.assert_allclose(
            dm["S1", "S2"],
            0.0,
            atol=1e-12,
            err_msg=f"{key}: identical samples must have distance 0",
        )


# ===========================================================================
# Known-value tests — simple 3-tip tree (hand-derived)
# ===========================================================================


def _simple_result() -> GUnifracResult:
    return gunifrac(_make_simple_ps(_SIMPLE_2S), alpha=(0, 0.5, 1))


def test_gunifrac_d1_known_value() -> None:
    np.testing.assert_allclose(
        _simple_result()["d_1"]["S1", "S2"], _EXPECT_D1, rtol=1e-6
    )


def test_gunifrac_d05_known_value() -> None:
    np.testing.assert_allclose(
        _simple_result()["d_0.5"]["S1", "S2"], _EXPECT_D05, rtol=1e-6
    )


def test_gunifrac_d0_known_value() -> None:
    np.testing.assert_allclose(
        _simple_result()["d_0"]["S1", "S2"], _EXPECT_D0, atol=1e-12
    )


def test_gunifrac_duw_known_value() -> None:
    np.testing.assert_allclose(
        gunifrac(_make_simple_ps(_SIMPLE_2S))["d_UW"]["S1", "S2"],
        _EXPECT_DUW,
        atol=1e-12,
    )


def test_gunifrac_dvaw_known_value() -> None:
    np.testing.assert_allclose(
        gunifrac(_make_simple_ps(_SIMPLE_2S))["d_VAW"]["S1", "S2"],
        _EXPECT_DVAW,
        rtol=1e-5,
    )


# ===========================================================================
# Known-value tests — Chen et al. (2012) 4-tip example
# ===========================================================================


def test_gunifrac_chen2012_d1_values() -> None:
    """d_1 for Chen et al. (2012) example: A–B=0.5, A–C=1.0, B–C=0.5."""
    dm = gunifrac(_make_chen_ps(), alpha=(1,))["d_1"]
    np.testing.assert_allclose(dm["SA", "SB"], 0.5, atol=1e-10)
    np.testing.assert_allclose(dm["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(dm["SB", "SC"], 0.5, atol=1e-10)


def test_gunifrac_chen2012_d0_values() -> None:
    """d_0 for Chen et al. (2012): A–B=2/3, A–C=1.0, B–C=2/3."""
    dm = gunifrac(_make_chen_ps(), alpha=(0,))["d_0"]
    np.testing.assert_allclose(dm["SA", "SB"], 2 / 3, atol=1e-10)
    np.testing.assert_allclose(dm["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(dm["SB", "SC"], 2 / 3, atol=1e-10)


def test_gunifrac_chen2012_duw_values() -> None:
    """d_UW (binarized) for Chen et al. (2012): A–B=3/5, A–C=1.0, B–C=3/5."""
    dm = gunifrac(_make_chen_ps())["d_UW"]
    np.testing.assert_allclose(dm["SA", "SB"], 3 / 5, atol=1e-10)
    np.testing.assert_allclose(dm["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(dm["SB", "SC"], 3 / 5, atol=1e-10)


def test_gunifrac_chen2012_d05_values() -> None:
    """d_0.5 for Chen et al. (2012): A–B≈0.5820, A–C=1.0, B–C≈0.5820."""
    dm = gunifrac(_make_chen_ps(), alpha=(0.5,))["d_0.5"]
    # R formula: num = sum(l·|diff|/sqrt(s)), den = sum(l·sqrt(s))
    # SA–SB: num = 3*sqrt(0.5) + 0.5/sqrt(1.5), den = 1 + 3*sqrt(0.5) + sqrt(1.5)
    expected_ab = (3 * np.sqrt(0.5) + 0.5 / np.sqrt(1.5)) / (
        1 + 3 * np.sqrt(0.5) + np.sqrt(1.5)
    )
    np.testing.assert_allclose(dm["SA", "SB"], expected_ab, rtol=1e-6)
    np.testing.assert_allclose(dm["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(dm["SB", "SC"], expected_ab, rtol=1e-6)


def test_gunifrac_chen2012_dvaw_values() -> None:
    """d_VAW for Chen et al. (2012): A–B=sqrt(5/3)/2, A–C=1.0, B–C=sqrt(5/3)/2."""
    dm = gunifrac(_make_chen_ps())["d_VAW"]
    expected_ab = np.sqrt(5.0 / 3.0) / 2.0
    np.testing.assert_allclose(dm["SA", "SB"], expected_ab, rtol=1e-6)
    np.testing.assert_allclose(dm["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(dm["SB", "SC"], expected_ab, rtol=1e-6)


# ===========================================================================
# Alpha ordering invariant: d_0 >= d_0.5 >= d_1 for partially-overlapping samples
# ===========================================================================


def test_gunifrac_alpha_ordering_chen2012() -> None:
    """For partially-overlapping samples, higher alpha yields lower distance."""
    result = gunifrac(_make_chen_ps(), alpha=(0, 0.5, 1))
    # SA and SB partially overlap; SA and SC are completely disjoint (all equal 1).
    d0 = result["d_0"]["SA", "SB"]
    d05 = result["d_0.5"]["SA", "SB"]
    d1 = result["d_1"]["SA", "SB"]
    assert d0 >= d05 - 1e-10
    assert d05 >= d1 - 1e-10


def test_gunifrac_alpha_ordering_simple() -> None:
    """d_0 >= d_0.5 >= d_1 also holds in the simple 3-tip fixture."""
    result = _simple_result()
    d0 = result["d_0"]["S1", "S2"]
    d05 = result["d_0.5"]["S1", "S2"]
    d1 = result["d_1"]["S1", "S2"]
    assert d0 >= d05 - 1e-10
    assert d05 >= d1 - 1e-10


# ===========================================================================
# d_VAW symmetry with d_0: completely disjoint samples give 1.0 for both
# ===========================================================================


def test_gunifrac_dvaw_disjoint_is_one() -> None:
    """d_VAW = 1.0 for completely disjoint samples, same as d_0 and d_1."""
    result = gunifrac(_make_chen_ps())
    # SA and SC share no OTUs; all variants should be exactly 1.0
    np.testing.assert_allclose(result["d_VAW"]["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(result["d_0"]["SA", "SC"], 1.0, atol=1e-10)
    np.testing.assert_allclose(result["d_1"]["SA", "SC"], 1.0, atol=1e-10)
