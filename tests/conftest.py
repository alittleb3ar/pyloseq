from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyloseq import OtuTable, Phyloseq, SampleData, TaxTable


@pytest.fixture
def simple_ps() -> Phyloseq:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.integers(0, 100, size=(3, 2)).astype(float),
        index=[f"OTU{i}" for i in range(3)],
        columns=[f"S{j}" for j in range(2)],
    )
    return Phyloseq(otu=OtuTable(df))


@pytest.fixture
def full_ps() -> Phyloseq:
    df_otu = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["OTU1", "OTU2"],
        columns=["S1", "S2"],
    )
    df_sam = pd.DataFrame(
        {"group": ["A", "B"], "depth": [100, 200]}, index=["S1", "S2"]
    )
    df_tax = pd.DataFrame(
        {"Phylum": ["Firm", "Bact"], "Genus": ["Lacto", "Bact"]},
        index=["OTU1", "OTU2"],
    )
    return Phyloseq(
        otu=OtuTable(df_otu),
        sam=SampleData(df_sam),
        tax=TaxTable(df_tax),
    )
