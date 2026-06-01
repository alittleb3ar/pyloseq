from pathlib import Path

from numpy.testing import assert_allclose

from pyloseq import PhyTree
from pyloseq.datasets import (load_esophagus_reference,
                              load_global_patterns_reference)

SIMPLE_NWK = "((OTU1:0.1,OTU2:0.2):0.3,OTU3:0.4);"


def test_from_newick() -> None:
    t = PhyTree.from_newick(SIMPLE_NWK)
    assert t.n_tips == 3
    assert set(t.tip_names) == {"OTU1", "OTU2", "OTU3"}


def test_total_branch_length() -> None:
    t = PhyTree.from_newick(SIMPLE_NWK)
    assert abs(t.total_branch_length - 1.0) < 1e-12


def test_is_rooted() -> None:
    rooted = PhyTree.from_newick("(A:1.0,B:1.0);")
    assert rooted.is_rooted
    unrooted = PhyTree.from_newick("(A:1.0,B:1.0,C:1.0);")
    assert not unrooted.is_rooted


def test_prune() -> None:
    t = PhyTree.from_newick(SIMPLE_NWK)
    pruned = t.prune(["OTU1", "OTU2"])
    assert pruned.n_tips == 2
    assert set(pruned.tip_names) == {"OTU1", "OTU2"}
    assert pruned.total_branch_length < t.total_branch_length


def test_newick_round_trip() -> None:
    t = PhyTree.from_newick(SIMPLE_NWK)
    nwk2 = t.to_newick()
    t2 = PhyTree.from_newick(nwk2)
    assert_allclose(t.total_branch_length, t2.total_branch_length, atol=1e-12)


def test_global_patterns_tree_ntips() -> None:
    ref = load_global_patterns_reference()
    t = PhyTree.from_newick(ref["phy_tree_newick"])
    assert t.n_tips == 19216


def test_esophagus_tree_ntips() -> None:
    ref = load_esophagus_reference()
    t = PhyTree.from_newick(ref["phy_tree_newick"])
    assert t.n_tips == 58


def test_eq_different_tip_sets() -> None:
    t1 = PhyTree.from_newick("(A:0.1,B:0.2);")
    t2 = PhyTree.from_newick("(A:0.1,C:0.2);")
    assert t1 != t2


def test_eq_same_newick() -> None:
    nwk = "((A:0.1,B:0.2):0.05,C:0.3);"
    assert PhyTree.from_newick(nwk) == PhyTree.from_newick(nwk)


def test_from_newick_file(tmp_path: Path) -> None:
    nwk_file = tmp_path / "tree.nwk"
    nwk_file.write_text(SIMPLE_NWK)
    t = PhyTree.from_newick_file(nwk_file)
    assert t.n_tips == 3
    assert set(t.tip_names) == {"OTU1", "OTU2", "OTU3"}


def test_copy_is_independent() -> None:
    t = PhyTree.from_newick(SIMPLE_NWK)
    t2 = t.copy()
    assert t is not t2
    assert t._tree is not t2._tree  # Newick round-trip must produce a fresh TreeNode
    t2._tree = PhyTree.from_newick("(A:0.5);")._tree  # replace copy's internals
    assert t.n_tips == 3  # original unaffected
