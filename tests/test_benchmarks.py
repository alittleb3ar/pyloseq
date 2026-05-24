"""Performance benchmarks for phyla — Ticket 6.1.

Run with:
    pytest tests/test_benchmarks.py --benchmark-only -v

Hard thresholds (CI fails on regression > 10%):
  - Phyloseq constructor on GlobalPatterns: < 3 s
  - weighted UniFrac on esophagus: < 10 s
  - tax_glom on GlobalPatterns at Family: < 5 s
  - rarefy_even_depth on GlobalPatterns: < 2 s
  - resident memory of GlobalPatterns Phyloseq: < 500 MB
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

GOLDEN_DIR = Path("tests/golden")
GP_PRESENT = (GOLDEN_DIR / "GlobalPatterns" / "otu_table.parquet").exists()
ES_PRESENT = (GOLDEN_DIR / "esophagus" / "otu_table.parquet").exists()

pytestmark = pytest.mark.skipif(
    not GP_PRESENT,
    reason="golden files not generated — run Rscript scripts/generate_golden.R",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def globalpatterns():
    from phyla import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable
    from phyla.testing.fixtures import load_global_patterns_reference

    ref = load_global_patterns_reference()
    return Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
        tax=TaxTable(ref["tax_table"]),
        tree=PhyTree.from_newick(ref["phy_tree_newick"]),
    )


@pytest.fixture(scope="module")
def esophagus():
    from phyla import OtuTable, Phyloseq, PhyTree
    from phyla.testing.fixtures import load_esophagus_reference

    ref = load_esophagus_reference()
    return Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        tree=PhyTree.from_newick(ref["phy_tree_newick"]) if "phy_tree_newick" in ref else None,
    )


# ---------------------------------------------------------------------------
# Ticket 6.1 — Benchmarks with hard thresholds
# ---------------------------------------------------------------------------

def test_benchmark_constructor(benchmark):
    """Phyloseq constructor on GlobalPatterns: < 3 s."""
    from phyla import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable
    from phyla.testing.fixtures import load_global_patterns_reference

    ref = load_global_patterns_reference()

    def build():
        return Phyloseq(
            otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
            sam=SampleData(ref["sample_data"]),
            tax=TaxTable(ref["tax_table"]),
            tree=PhyTree.from_newick(ref["phy_tree_newick"]),
        )

    result = benchmark.pedantic(build, rounds=3, iterations=1)
    assert result is not None
    assert benchmark.stats["mean"] < 3.0, (
        f"Constructor too slow: {benchmark.stats['mean']:.2f}s (threshold: 3s)"
    )


def test_benchmark_tax_glom_family(benchmark, globalpatterns):
    """tax_glom at Family rank on GlobalPatterns: < 5 s."""
    from phyla import tax_glom

    result = benchmark.pedantic(
        tax_glom, args=(globalpatterns, "Family"), rounds=3, iterations=1
    )
    assert result is not None
    assert benchmark.stats["mean"] < 5.0, (
        f"tax_glom too slow: {benchmark.stats['mean']:.2f}s (threshold: 5s)"
    )


def test_benchmark_rarefy(benchmark, globalpatterns):
    """rarefy_even_depth on GlobalPatterns: < 2 s."""
    from phyla import rarefy_even_depth

    result = benchmark.pedantic(
        rarefy_even_depth,
        args=(globalpatterns,),
        kwargs={"rng_seed": 42},
        rounds=3,
        iterations=1,
    )
    assert result is not None
    assert benchmark.stats["mean"] < 2.0, (
        f"rarefy_even_depth too slow: {benchmark.stats['mean']:.2f}s (threshold: 2s)"
    )


def test_benchmark_bray_distance(benchmark, globalpatterns):
    """Bray-Curtis distance on GlobalPatterns: < 5 s."""
    result = benchmark.pedantic(
        globalpatterns.distance,
        args=("bray",),
        rounds=3,
        iterations=1,
    )
    assert result is not None
    assert benchmark.stats["mean"] < 5.0, (
        f"distance(bray) too slow: {benchmark.stats['mean']:.2f}s (threshold: 5s)"
    )


def test_benchmark_pcoa(benchmark, globalpatterns):
    """PCoA on GlobalPatterns Bray-Curtis: < 5 s."""
    from phyla import distance

    dm = distance(globalpatterns, "bray")

    result = benchmark.pedantic(
        globalpatterns.ordinate,
        args=("PCoA",),
        kwargs={"distance": dm},
        rounds=3,
        iterations=1,
    )
    assert result is not None
    assert benchmark.stats["mean"] < 5.0, (
        f"PCoA too slow: {benchmark.stats['mean']:.2f}s (threshold: 5s)"
    )


@pytest.mark.skipif(not ES_PRESENT, reason="esophagus golden files not generated")
def test_benchmark_unifrac_weighted(benchmark, esophagus):
    """Weighted UniFrac on esophagus: < 10 s."""
    from phyla import unifrac

    result = benchmark.pedantic(
        unifrac,
        args=(esophagus,),
        kwargs={"weighted": True},
        rounds=3,
        iterations=1,
    )
    assert result is not None
    assert benchmark.stats["mean"] < 10.0, (
        f"weighted UniFrac too slow: {benchmark.stats['mean']:.2f}s (threshold: 10s)"
    )


def test_benchmark_memory_globalpatterns():
    """GlobalPatterns resident object: < 500 MB."""
    from phyla import OtuTable, Phyloseq, PhyTree, SampleData, TaxTable
    from phyla.testing.fixtures import load_global_patterns_reference

    ref = load_global_patterns_reference()

    tracemalloc.start()
    ps = Phyloseq(
        otu=OtuTable(ref["otu_table"], taxa_are_rows=True),
        sam=SampleData(ref["sample_data"]),
        tax=TaxTable(ref["tax_table"]),
        tree=PhyTree.from_newick(ref["phy_tree_newick"]),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    assert peak_mb < 500, (
        f"Memory too high: {peak_mb:.1f} MB (threshold: 500 MB)"
    )
    assert ps is not None
