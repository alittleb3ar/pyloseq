from __future__ import annotations

import importlib
from typing import Any

import pandas as pd
import pytest

from pyloseq.datasets import (_load_dataset, load_enterotype_reference,
                              load_esophagus_reference,
                              load_global_patterns_reference, load_golden,
                              load_soilrep_reference)


def test_testing_module_importable() -> None:
    mod = importlib.import_module("pyloseq.datasets")
    for name in [
        "load_global_patterns_reference",
        "load_enterotype_reference",
        "load_esophagus_reference",
        "load_soilrep_reference",
        "load_golden",
    ]:
        assert hasattr(mod, name), f"pyloseq.testing missing: {name}"


def test_load_missing_dataset_raises() -> None:

    with pytest.raises(FileNotFoundError):
        _load_dataset("nonexistent_dataset_xyz")


def test_global_patterns_loads(benchmark: Any) -> None:

    result = benchmark(load_global_patterns_reference)
    assert "otu_table" in result
    assert "taxa_sums" in result
    assert "sample_sums" in result


def test_global_patterns_otu_table_shape() -> None:

    ref = load_global_patterns_reference()
    # GlobalPatterns: 19216 taxa × 26 samples, taxa as rows
    otu = ref["otu_table"]
    assert otu.shape == (19216, 26), f"Unexpected OTU table shape: {otu.shape}"


def test_global_patterns_tax_table_shape() -> None:

    ref = load_global_patterns_reference()
    assert ref["tax_table"].shape == (19216, 7)


def test_enterotype_sample_data_shape() -> None:

    ref = load_enterotype_reference()
    # enterotype: 280 samples × 9 variables
    assert ref["sample_data"].shape == (280, 9)


def test_esophagus_has_tree() -> None:

    ref = load_esophagus_reference()
    assert "phy_tree_newick" in ref
    assert len(ref["phy_tree_newick"]) > 10


def test_taxa_sums_is_series() -> None:

    ref = load_global_patterns_reference()
    assert isinstance(ref["taxa_sums"], pd.Series)
    assert len(ref["taxa_sums"]) == 19216


def test_load_golden_estimate_richness() -> None:

    df = load_golden("GlobalPatterns", "estimate_richness")
    assert "Observed" in df.columns or "Shannon" in df.columns


# ---------------------------------------------------------------------------
# soilrep golden file tests
# ---------------------------------------------------------------------------


def test_soilrep_loads() -> None:
    ref = load_soilrep_reference()
    assert "otu_table" in ref


def test_soilrep_otu_table_shape() -> None:
    ref = load_soilrep_reference()
    # soilrep: 16,825 taxa × 56 samples
    assert ref["otu_table"].shape == (16825, 56)


def test_soilrep_sample_data_shape() -> None:
    ref = load_soilrep_reference()
    # 56 samples with treatment metadata
    assert ref["sample_data"].shape[0] == 56


def test_soilrep_taxa_sums_is_series() -> None:
    ref = load_soilrep_reference()
    assert isinstance(ref["taxa_sums"], pd.Series)
    assert len(ref["taxa_sums"]) == 16825


def test_soilrep_sample_sums_nonnegative() -> None:
    ref = load_soilrep_reference()
    assert (ref["sample_sums"] >= 0).all()


def test_soilrep_estimate_richness_golden() -> None:
    """estimate_richness output matches R phyloseq reference for soilrep."""
    import numpy as np

    from pyloseq import OtuTable, Phyloseq
    from pyloseq._diversity import estimate_richness

    ref = load_soilrep_reference()
    golden = load_golden("soilrep", "estimate_richness")

    ps = Phyloseq(otu=OtuTable(ref["otu_table"]))

    result = estimate_richness(ps, measures=["Observed", "Chao1", "Shannon"])

    common_samples = result.index.intersection(golden.index)
    assert len(common_samples) > 0

    if "Observed" in golden.columns:
        np.testing.assert_allclose(
            result.loc[common_samples, "Observed"].values,
            golden.loc[common_samples, "Observed"].values,
            rtol=1e-3,
            err_msg="Observed richness does not match R reference for soilrep",
        )

    if "Shannon" in golden.columns:
        np.testing.assert_allclose(
            result.loc[common_samples, "Shannon"].values,
            golden.loc[common_samples, "Shannon"].values,
            atol=1e-4,
            err_msg="Shannon diversity does not match R reference for soilrep",
        )
