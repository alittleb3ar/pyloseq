"""Private numerical and structural helpers shared across computation modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pyloseq._exceptions import pyloseqValidationError

if TYPE_CHECKING:
    from pyloseq._tree import PhyTree


def _row_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each row by its sum; rows that sum to zero become all-zero."""
    row_sums = df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return df.div(row_sums, axis=0)


def _filter_otu_to_tree(otu_df: pd.DataFrame, tree: PhyTree) -> pd.DataFrame:
    """Return *otu_df* restricted to taxa present in *tree*.

    Parameters
    ----------
    otu_df:
        Samples-as-rows OTU DataFrame (columns = taxa).
    tree:
        Phylogenetic tree; taxa not in ``tree.tip_names`` are dropped.

    Raises
    ------
    pyloseqValidationError
        If no taxa remain after filtering.
    """
    tree_tips = set(tree.tip_names)
    taxa = [t for t in otu_df.columns if t in tree_tips]
    if not taxa:
        raise pyloseqValidationError(
            "No taxa names match tree tip labels. "
            "Check that taxa_names and tree tip names are consistent."
        )
    return otu_df[taxa]
