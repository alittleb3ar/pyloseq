"""Phylogenetic tree container wrapping scikit-bio TreeNode."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import skbio
import skbio.tree


class PhyTree:
    """Wraps a ``skbio.tree.TreeNode`` with a phyloseq-compatible interface.

    R reference: phyloseq::phy_tree(object)
    """

    def __init__(self, tree_node: skbio.tree.TreeNode) -> None:
        if not isinstance(tree_node, skbio.tree.TreeNode):
            raise TypeError(
                f"PhyTree requires a skbio.tree.TreeNode, got {type(tree_node)!r}"
            )
        self._tree = tree_node

    @classmethod
    def from_newick(cls, s: str) -> PhyTree:
        """Construct from a Newick string.

        R reference: phy_tree(read.tree(text=s))
        """
        # convert_underscores=False: preserve underscores in tip names as-is.
        # Newick convention maps underscores to spaces, but phyloseq datasets
        # use underscores as part of OTU identifiers — we must not mangle them.
        tree = skbio.tree.TreeNode.read(
            StringIO(s), format="newick", convert_underscores=False
        )
        return cls(tree)

    @classmethod
    def from_newick_file(cls, path: str | Path) -> PhyTree:
        """Construct from a Newick file on disk.

        R reference: phy_tree(read.tree(file=path))
        """
        tree = skbio.tree.TreeNode.read(
            str(path), format="newick", convert_underscores=False
        )
        return cls(tree)

    @classmethod
    def from_ape_rds(cls, path: str | Path) -> PhyTree:
        """Construct from an R ``phylo`` object serialized as ``.rds``.

        Requires ``pyreadr`` (``pip install pyreadr``).

        R reference: readRDS(path)
        """
        raise NotImplementedError(
            "from_ape_rds() is not yet implemented. "
            "Export the tree as Newick from R with ape::write.tree() and use from_newick_file()."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tip_names(self) -> list[str]:
        """Names of all leaf nodes.

        R reference: taxa_names(phy_tree(x))
        """
        return [t.name for t in self._tree.tips() if t.name is not None]

    @property
    def internal_names(self) -> list[str]:
        """Names of all internal (non-tip) nodes, excluding the root if unnamed.

        R reference: phy_tree(x)$node.label
        """
        return [
            n.name
            for n in self._tree.traverse()
            if not n.is_tip() and n.name is not None
        ]

    @property
    def n_tips(self) -> int:
        """Number of tip (leaf) nodes.

        R reference: ntaxa(phy_tree(x))
        """
        return sum(1 for _ in self._tree.tips())

    @property
    def total_branch_length(self) -> float:
        """Sum of all branch lengths in the tree.

        R reference: sum(phy_tree(x)$edge.length)
        """
        return float(
            sum(n.length for n in self._tree.traverse() if n.length is not None)
        )

    @property
    def is_rooted(self) -> bool:
        """``True`` if the root has exactly 2 children (bifurcating root).

        R reference: is.rooted(phy_tree(x))
        """
        return len(list(self._tree.children)) == 2

    # ------------------------------------------------------------------
    # Manipulation
    # ------------------------------------------------------------------

    def prune(self, keep: list[str]) -> PhyTree:
        """Return a new tree containing only the specified tips and their ancestors.

        Equivalent to ``ape::drop.tip`` with the complement set.

        R reference: prune_taxa(keep, ps) (on the tree component)
        """
        pruned = self._tree.shear(set(keep))
        return PhyTree(pruned)

    def to_newick(self) -> str:
        """Serialize to a Newick string.

        R reference: ape::write.tree(phy_tree(x))
        """
        buf = StringIO()
        self._tree.write(buf, format="newick")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        rooted = "rooted" if self.is_rooted else "unrooted"
        return f"PhyTree({self.n_tips} tips, {rooted})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhyTree):
            return NotImplemented
        # Fast early-exit on tip sets and total branch length before full Newick compare
        if set(self.tip_names) != set(other.tip_names):
            return False
        if abs(self.total_branch_length - other.total_branch_length) >= 1e-10:
            return False
        return self.to_newick() == other.to_newick()

    def copy(self) -> PhyTree:
        """Return a deep copy of this PhyTree via Newick round-trip."""
        return PhyTree.from_newick(self.to_newick())
