"""Data manipulation functions mirroring R phyloseq's pre-processing API.

All functions return new ``Phyloseq`` objects; inputs are never mutated.

R reference: phyloseq vignette "Preprocessing"
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from pyloseq._exceptions import pyloseqValidationError
from pyloseq._otu_table import OtuTable
from pyloseq._refseq import RefSeq
from pyloseq._sample_data import SampleData
from pyloseq._tax_table import TaxTable
from pyloseq._tree import PhyTree

if TYPE_CHECKING:
    from pyloseq._phyloseq import Phyloseq

_T = TypeVar("_T")

_SENTINEL = object()  # private sentinel: "copy this component from the source Phyloseq"


def _keep(val: Any, default: _T) -> _T:
    """Return *default* when *val* is the sentinel, otherwise return *val* as-is."""
    return default if val is _SENTINEL else val


def _rebuild_ps(
    ps: Phyloseq,
    otu: OtuTable,
    *,
    sam: SampleData | None | object = _SENTINEL,
    tax: TaxTable | None | object = _SENTINEL,
    tree: PhyTree | None | object = _SENTINEL,
    refseq: RefSeq | None | object = _SENTINEL,
    metadata: dict[str, Any] | object = _SENTINEL,
) -> Phyloseq:
    """Construct a new Phyloseq, copying unchanged components from *ps* by default."""
    from pyloseq._phyloseq import Phyloseq as _Phyloseq  # noqa: PLC0415

    return _Phyloseq(
        otu=otu,
        sam=_keep(sam, ps.sample_data.copy() if ps.sample_data is not None else None),
        tax=_keep(tax, ps.tax_table.copy() if ps.tax_table is not None else None),
        tree=_keep(tree, ps.phy_tree.copy() if ps.phy_tree is not None else None),
        refseq=_keep(refseq, ps.refseq.copy() if ps.refseq is not None else None),
        metadata=_keep(metadata, dict(ps.metadata)),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_refseq(ps: Phyloseq, keep_taxa: list[str]) -> RefSeq | None:
    """Return a RefSeq filtered to ``keep_taxa``, or ``None`` if ps has no refseq."""
    if ps.refseq is None:
        return None
    import skbio  # noqa: PLC0415

    ref_names = set(ps.refseq.taxa_names)
    new_seqs: dict[str, skbio.DNA] = {
        k: ps.refseq[k] for k in keep_taxa if k in ref_names
    }
    return RefSeq(new_seqs) if new_seqs else None


def _ps_copy(ps: Phyloseq) -> Phyloseq:
    """Shallow-copy a Phyloseq, deep-copying each component."""
    from pyloseq._phyloseq import Phyloseq as _Phyloseq

    return _Phyloseq(
        otu=ps.otu_table.copy(),
        sam=ps.sample_data.copy() if ps.sample_data is not None else None,
        tax=ps.tax_table.copy() if ps.tax_table is not None else None,
        tree=ps.phy_tree.copy() if ps.phy_tree is not None else None,
        refseq=ps.refseq.copy() if ps.refseq is not None else None,
        metadata=dict(ps.metadata),
    )


def _otu_taxa_rows(ps: Phyloseq) -> pd.DataFrame:
    """Return OTU table as a DataFrame with taxa as rows (taxa × samples)."""
    df = ps.otu_table.to_dataframe()
    if not ps.otu_table.taxa_are_rows:
        df = df.T
    return df


def _otu_samples_rows(ps: Phyloseq) -> pd.DataFrame:
    """Return OTU table as a DataFrame with samples as rows (samples × taxa)."""
    df = ps.otu_table.to_dataframe()
    if ps.otu_table.taxa_are_rows:
        df = df.T
    return df


# ---------------------------------------------------------------------------
# Ticket 3.2 — prune_samples, prune_taxa
# ---------------------------------------------------------------------------


def prune_taxa(
    names: list[str] | pd.Index,
    ps: Phyloseq,
) -> Phyloseq:
    """Return a new Phyloseq containing only the specified taxa.

    Parameters
    ----------
    names:
        Taxa to keep. Order is preserved; names absent from ``ps`` are ignored.
    ps:
        Source ``Phyloseq`` object.

    R reference: prune_taxa(taxa, x)
    """
    taxa_set = set(ps.taxa_names)
    keep = [n for n in names if n in taxa_set]
    keep_idx = pd.Index(keep)

    df = _otu_taxa_rows(ps)
    new_otu = OtuTable(df.reindex(keep_idx), taxa_are_rows=True)

    new_tax = None
    if ps.tax_table is not None:
        new_tax = TaxTable(ps.tax_table.to_frame().reindex(keep_idx))

    new_tree = None
    if ps.phy_tree is not None:
        tree_keep = [t for t in keep if t in set(ps.phy_tree.tip_names)]
        if len(tree_keep) >= 2:
            new_tree = ps.phy_tree.prune(tree_keep)

    return _rebuild_ps(
        ps, new_otu, tax=new_tax, tree=new_tree, refseq=_filter_refseq(ps, keep)
    )


def prune_samples(
    names: list[str] | pd.Index,
    ps: Phyloseq,
) -> Phyloseq:
    """Return a new Phyloseq containing only the specified samples.

    Parameters
    ----------
    names:
        Samples to keep. Order is preserved; names absent from ``ps`` are ignored.
    ps:
        Source ``Phyloseq`` object.

    R reference: prune_samples(samples, x)
    """
    sample_set = set(ps.sample_names)
    keep = [n for n in names if n in sample_set]
    keep_idx = pd.Index(keep)

    df = _otu_taxa_rows(ps)
    new_otu = OtuTable(df.reindex(columns=keep_idx), taxa_are_rows=True)

    new_sam = None
    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        new_sam = SampleData(sam_df.reindex(keep_idx))

    return _rebuild_ps(ps, new_otu, sam=new_sam)


# ---------------------------------------------------------------------------
# Ticket 3.1 — subset_samples, subset_taxa
# ---------------------------------------------------------------------------


def subset_samples(
    ps: Phyloseq,
    expr: Callable[..., Any] | str,
) -> Phyloseq:
    """Return a new Phyloseq with samples matching a filter expression.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object (must have ``sample_data``).
    expr:
        Either a callable applied row-wise to ``sample_data`` returning a bool,
        or a pandas-query string evaluated against ``sample_data``.

    Notes
    -----
    For the string form, ``sample_data`` is evaluated with
    :meth:`pandas.DataFrame.query`. The sample ID is the index, so reference it
    as ``index`` (e.g. ``'index == "S1"'``), and column names containing spaces
    (such as ``"Consensus Lineage"``) must be wrapped in backticks. Use the
    callable form to avoid these quoting rules.

    Examples
    --------
    >>> subset_samples(ps, lambda s: s["SampleType"] == "Soil")
    >>> subset_samples(ps, 'SampleType == "Soil"')

    R reference: subset_samples(physeq, ...)
    """
    if ps.sample_data is None:
        raise pyloseqValidationError("subset_samples requires sample_data")

    sam_df = ps.sample_data.to_frame()

    if callable(expr):
        mask: pd.Series = sam_df.apply(expr, axis=1)
        keep = list(sam_df.index[mask.astype(bool)])
    elif isinstance(expr, str):
        keep = list(sam_df.query(expr).index)
    else:
        raise TypeError(f"expr must be callable or str, got {type(expr)!r}")

    return prune_samples(keep, ps)


def subset_taxa(
    ps: Phyloseq,
    expr: Callable[..., Any] | str,
) -> Phyloseq:
    """Return a new Phyloseq with taxa matching a filter expression.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object (must have ``tax_table``).
    expr:
        Either a callable applied row-wise to ``tax_table`` returning a bool,
        or a pandas-query string evaluated against ``tax_table``.

    Notes
    -----
    For the string form, rank columns with spaces (e.g. ``"Consensus Lineage"``)
    must be backtick-quoted for :meth:`pandas.DataFrame.query`. Use the callable
    form to avoid these quoting rules.

    Examples
    --------
    >>> subset_taxa(ps, lambda t: t["Phylum"] == "Chlamydiae")
    >>> subset_taxa(ps, 'Phylum == "Chlamydiae"')

    R reference: subset_taxa(physeq, ...)
    """
    if ps.tax_table is None:
        raise pyloseqValidationError("subset_taxa requires tax_table")

    tax_df = ps.tax_table.to_frame()

    if callable(expr):
        mask = tax_df.apply(expr, axis=1)
        keep = list(tax_df.index[mask.astype(bool)])
    elif isinstance(expr, str):
        keep = list(tax_df.query(expr).index)
    else:
        raise TypeError(f"expr must be callable or str, got {type(expr)!r}")

    return prune_taxa(keep, ps)


# ---------------------------------------------------------------------------
# Ticket 3.3 — filter_taxa, kOverA
# ---------------------------------------------------------------------------


def kOverA(k: int, A: float) -> Callable[[pd.Series], bool]:
    """Return a predicate: True if >= k samples have abundance > A.

    This is a closure factory matching R's ``genefilter::kOverA``.

    Parameters
    ----------
    k:
        Minimum number of samples that must exceed threshold.
    A:
        Abundance threshold.

    R reference: kOverA(k, A)
    """

    def _predicate(x: pd.Series) -> bool:
        return int((x > A).sum()) >= k

    return _predicate


def filter_taxa(
    ps: Phyloseq,
    predicate: Callable[[pd.Series], bool],
) -> Phyloseq:
    """Return a new Phyloseq containing only taxa that satisfy ``predicate``.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object.
    predicate:
        A callable accepting a ``pd.Series`` of abundances across all samples
        for one taxon, returning ``True`` to keep, ``False`` to drop.

    Notes
    -----
    This corresponds to R's ``filter_taxa(physeq, flist, prune=TRUE)``. For the
    ``prune=FALSE`` behaviour (return the boolean mask without pruning), use
    :func:`taxa_filter_mask`.

    R reference: filter_taxa(physeq, flist, prune=TRUE)
    """
    df = _otu_taxa_rows(ps)
    keep_mask: pd.Series = df.apply(predicate, axis=1)
    keep = list(keep_mask.index[keep_mask])
    return prune_taxa(keep, ps)


def taxa_filter_mask(
    ps: Phyloseq,
    predicate: Callable[[pd.Series], bool],
) -> pd.Series:
    """Return a boolean ``pd.Series`` (indexed by taxa) for a per-taxon predicate.

    Use this when you want to inspect the mask before pruning.
    To obtain a pruned ``Phyloseq`` directly, use :func:`filter_taxa`.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object.
    predicate:
        A callable accepting a ``pd.Series`` of abundances across all samples
        for one taxon, returning ``True`` to keep, ``False`` to drop.

    R reference: filter_taxa(physeq, flist, prune=FALSE)
    """
    df = _otu_taxa_rows(ps)
    return df.apply(predicate, axis=1)


# ---------------------------------------------------------------------------
# Ticket 3.4 — transform_sample_counts
# ---------------------------------------------------------------------------


def transform_sample_counts(
    ps: Phyloseq,
    fn: Callable[[pd.Series], pd.Series],
) -> Phyloseq:
    """Apply a per-sample transformation to the abundance table.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object.
    fn:
        A callable accepting a ``pd.Series`` of abundances for one sample
        (indexed by taxa name), returning a transformed series of equal length.

    Examples
    --------
    >>> transform_sample_counts(ps, lambda x: x / x.sum())

    R reference: transform_sample_counts(physeq, function(x) x / sum(x))
    """
    df = _otu_taxa_rows(ps)
    new_otu = OtuTable(df.apply(fn, axis=0), taxa_are_rows=True)
    return _rebuild_ps(ps, new_otu)


# ---------------------------------------------------------------------------
# Ticket 3.5 — rarefy_even_depth
# ---------------------------------------------------------------------------


def rarefy_even_depth(
    ps: Phyloseq,
    sample_size: int | None = None,
    rng_seed: int | None = 42,
    replace: bool = True,
    trim_otus: bool = True,
    verbose: bool = True,
    compat: str | None = None,
) -> Phyloseq:
    """Rarefy all samples to even sequencing depth by subsampling.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object.
    sample_size:
        Target depth. Defaults to ``min(sample_sums(ps))``.
    rng_seed:
        Seed for ``numpy.random.default_rng``. Pass ``None`` for
        non-reproducible draws.
    replace:
        If ``True`` (default), use multinomial sampling (with replacement).
        If ``False``, sample without replacement from the read pool. Note that
        the without-replacement path materialises a read pool of size equal to
        each sample's depth, which can be large for very deep samples.
    trim_otus:
        Remove taxa that are zero in all samples after rarefaction.
    verbose:
        Emit a warning when samples are dropped.
    compat:
        Reserved for future R-compatible RNG mode; currently ignored.

    R reference: rarefy_even_depth(physeq, sample.size, rngseed, replace, trimOTUs, verbose)
    """
    if compat is not None and compat != "r-vegan":
        raise ValueError(f"Unknown compat mode: {compat!r}. Use 'r-vegan' or None.")

    df = _otu_taxa_rows(ps)
    ss = df.sum(axis=0)

    if sample_size is None:
        sample_size = int(ss.min())

    if sample_size <= 0:
        raise ValueError(f"sample_size must be > 0, got {sample_size!r}")

    # Drop samples below threshold
    drop_mask = ss < sample_size
    if drop_mask.any():
        n_drop = int(drop_mask.sum())
        if verbose:
            warnings.warn(
                f"Dropping {n_drop} sample(s) with fewer than {sample_size} reads.",
                stacklevel=2,
            )
        df = df.loc[:, ~drop_mask]

    if df.shape[1] == 0:
        raise pyloseqValidationError(
            f"All samples have fewer than {sample_size} reads. "
            "Lower sample_size or check your data."
        )

    rng = np.random.default_rng(rng_seed)

    rarefied_cols: dict[str, np.ndarray] = {}
    for col in df.columns:
        counts = np.asarray(df[col], dtype=int)
        total = int(counts.sum())
        if replace:
            prob = counts / total
            new_counts = rng.multinomial(sample_size, prob).astype(float)
        else:
            pool = np.repeat(np.arange(len(counts)), counts)
            chosen = rng.choice(pool, size=sample_size, replace=False)
            new_counts = np.bincount(chosen, minlength=len(counts)).astype(float)
        rarefied_cols[str(col)] = new_counts

    new_otu_df = pd.DataFrame(rarefied_cols, index=df.index)

    if trim_otus:
        nonzero = new_otu_df.sum(axis=1) > 0
        new_otu_df = new_otu_df.loc[nonzero]

    surviving_taxa = list(new_otu_df.index)
    surviving_samples = list(new_otu_df.columns)

    # Build a Phyloseq with the rarefied counts, pruned to survivors
    new_otu = OtuTable(new_otu_df, taxa_are_rows=True)

    # Sample data: keep surviving samples
    new_sam = None
    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        new_sam = SampleData(
            sam_df.reindex(pd.Index(surviving_samples)).dropna(how="all")
        )

    # Tax table: keep surviving taxa
    new_tax = None
    if ps.tax_table is not None:
        tax_df = ps.tax_table.to_frame()
        new_tax = TaxTable(tax_df.reindex(pd.Index(surviving_taxa)))

    # Tree: keep surviving taxa
    new_tree = None
    if ps.phy_tree is not None:
        _tip_set = set(ps.phy_tree.tip_names)
        tree_keep = [t for t in surviving_taxa if t in _tip_set]
        if len(tree_keep) >= 2:
            new_tree = ps.phy_tree.prune(tree_keep)

    return _rebuild_ps(
        ps,
        new_otu,
        sam=new_sam,
        tax=new_tax,
        tree=new_tree,
        refseq=_filter_refseq(ps, surviving_taxa),
    )


# ---------------------------------------------------------------------------
# Ticket 3.8 — merge_taxa
# ---------------------------------------------------------------------------


def merge_taxa(
    ps: Phyloseq,
    eqtaxa: list[str],
    archetype: str | None = None,
) -> Phyloseq:
    """Merge a set of taxa into a single representative.

    Abundances are summed; the archetype's taxonomy row is retained.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object.
    eqtaxa:
        Taxa to merge.
    archetype:
        The taxon whose metadata row is retained. Defaults to the most-abundant
        member (sum across all samples).

    R reference: merge_taxa(x, eqtaxa, archetype)
    """
    df = _otu_taxa_rows(ps)
    present = [t for t in eqtaxa if t in df.index]
    if len(present) < 2:
        return _ps_copy(ps)

    if archetype is None:
        archetype = str(df.loc[present].sum(axis=1).idxmax())

    merged_row: pd.Series = df.loc[present].sum(axis=0)

    non_eq = [t for t in df.index if t not in set(present)]
    new_otu_df = df.loc[non_eq].copy()
    new_otu_df.loc[archetype] = merged_row
    new_otu = OtuTable(new_otu_df, taxa_are_rows=True)

    new_tax = None
    if ps.tax_table is not None:
        tax_df = ps.tax_table.to_frame()
        keep_tax = [t for t in new_otu_df.index if t in tax_df.index]
        new_tax = TaxTable(tax_df.loc[keep_tax])

    # Tree: prune to surviving taxa
    new_tree = None
    if ps.phy_tree is not None:
        tree_tips = set(ps.phy_tree.tip_names)
        tree_keep = [t for t in new_otu_df.index if t in tree_tips]
        if len(tree_keep) >= 2:
            new_tree = ps.phy_tree.prune(tree_keep)

    surviving_taxa_merge = list(new_otu_df.index)
    return _rebuild_ps(
        ps,
        new_otu,
        tax=new_tax,
        tree=new_tree,
        refseq=_filter_refseq(ps, surviving_taxa_merge),
    )


# ---------------------------------------------------------------------------
# Ticket 3.6 — tax_glom
# ---------------------------------------------------------------------------


def tax_glom(
    ps: Phyloseq,
    taxrank: str,
    na_rm: bool = True,
    bad_empty: tuple[Any, ...] = (None, "", " ", "\t"),
) -> Phyloseq:
    """Agglomerate taxa to a specified taxonomic rank.

    Taxa that share the same value at ``taxrank`` (and all coarser ranks) are
    summed; the most-abundant member's row is kept as the archetype.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object (must have ``tax_table``).
    taxrank:
        Rank to agglomerate at (e.g. ``"Genus"``).
    na_rm:
        Drop taxa whose value at ``taxrank`` is ``None``, empty, or
        in ``bad_empty``.
    bad_empty:
        Values treated as missing at ``taxrank``.

    R reference: tax_glom(physeq, taxrank, NArm, bad_empty)
    """
    if ps.tax_table is None:
        raise pyloseqValidationError("tax_glom requires a TaxTable")

    tax_df = ps.tax_table.to_frame()
    ranks = list(tax_df.columns)

    if taxrank not in ranks:
        raise pyloseqValidationError(
            f"Rank '{taxrank}' not found in tax_table. Available: {ranks}"
        )

    rank_idx = ranks.index(taxrank)
    coarser = ranks[: rank_idx + 1]  # all ranks up to and including taxrank

    otu_df = _otu_taxa_rows(ps)

    # Align OTU and tax tables to their intersection
    common_taxa = otu_df.index.intersection(tax_df.index)
    otu_df = otu_df.reindex(common_taxa)
    tax_df = tax_df.reindex(common_taxa)

    # Remove taxa with bad/NA at the target rank
    if na_rm:
        target_vals = tax_df[taxrank]
        bad_set = set(bad_empty)
        mask = target_vals.apply(lambda v: not (pd.isna(v) or v in bad_set))
        otu_df = otu_df[mask]
        tax_df = tax_df[mask]

    if len(otu_df) == 0:
        raise pyloseqValidationError(
            f"No taxa remaining after na_rm filtering at rank '{taxrank}'."
        )

    # Build a grouping key from the coarser ranks (fill NA for stable grouping)
    group_key_ser: pd.Series = tax_df[coarser].apply(
        lambda row: tuple(
            str(v) if not (isinstance(v, float) and np.isnan(v)) else "__NA__"
            for v in row
        ),
        axis=1,
    )

    # Per-taxon sums for archetype selection
    taxa_sum: pd.Series = otu_df.sum(axis=1)

    archetype_for_key: dict[tuple[Any, ...], str] = {}
    new_rows: dict[str, pd.Series] = {}

    for key, group_indices in group_key_ser.groupby(group_key_ser):
        group_taxa = list(group_indices.index)
        archetype = str(taxa_sum.loc[group_taxa].idxmax())
        archetype_for_key[key] = archetype
        new_rows[archetype] = otu_df.loc[group_taxa].sum(axis=0)

    archetypes = list(new_rows.keys())
    new_otu_df = pd.DataFrame(new_rows).T
    new_otu_df.index.name = None

    new_otu = OtuTable(new_otu_df, taxa_are_rows=True)
    new_tax = TaxTable(tax_df.loc[archetypes])

    # Tree: prune to archetypes
    new_tree = None
    if ps.phy_tree is not None:
        tree_tips = set(ps.phy_tree.tip_names)
        tree_keep = [t for t in archetypes if t in tree_tips]
        if len(tree_keep) >= 2:
            new_tree = ps.phy_tree.prune(tree_keep)

    return _rebuild_ps(
        ps, new_otu, tax=new_tax, tree=new_tree, refseq=_filter_refseq(ps, archetypes)
    )


# ---------------------------------------------------------------------------
# Ticket 3.7 — tip_glom
# ---------------------------------------------------------------------------


def tip_glom(
    ps: Phyloseq,
    h: float = 0.2,
    hcfun: str = "average",
) -> Phyloseq:
    """Agglomerate taxa by phylogenetic distance.

    Hierarchical clustering on pairwise patristic distances groups tips whose
    within-cluster distance is <= ``h``; each group is merged via
    :func:`merge_taxa`.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object (must have ``phy_tree``).
    h:
        Height cutoff for :func:`scipy.cluster.hierarchy.fcluster`.
    hcfun:
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`
        (e.g. ``"average"``, ``"complete"``, ``"ward"``).

    R reference: tip_glom(physeq, h, hcfun)
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if ps.phy_tree is None:
        raise pyloseqValidationError("tip_glom requires a PhyTree")

    tree_node = ps.phy_tree._tree
    dm = tree_node.tip_tip_distances()
    dm_ids = list(dm.ids)

    # Restrict to taxa that are also in the OTU table
    otu_taxa = set(ps.taxa_names)
    filtered_ids = [t for t in dm_ids if t in otu_taxa]
    if len(filtered_ids) < 2:
        return _ps_copy(ps)

    idx = [dm_ids.index(t) for t in filtered_ids]
    sub_matrix: np.ndarray = dm.data[np.ix_(idx, idx)]

    condensed = squareform(sub_matrix, checks=False)
    Z = linkage(condensed, method=hcfun)
    labels = fcluster(Z, t=h, criterion="distance")

    df = _otu_taxa_rows(ps)

    from collections import defaultdict  # noqa: PLC0415

    # Group taxa by cluster assignment; for multi-member clusters the archetype
    # is the most-abundant member. Singletons keep their own name (the default
    # in the .get(t, t) lookup below), so no placeholder pass is needed.
    cluster_groups: dict[int, list[str]] = defaultdict(list)
    for taxon, cluster_id in zip(filtered_ids, labels, strict=False):
        cluster_groups[int(cluster_id)].append(taxon)

    taxon_to_archetype: dict[str, str] = {}
    for taxa_in_cluster in cluster_groups.values():
        present = [t for t in taxa_in_cluster if t in df.index]
        if len(present) > 1:
            archetype = str(df.loc[present].sum(axis=1).idxmax())
            for t in present:
                taxon_to_archetype[t] = archetype

    # Single-pass groupby-sum: assign archetype label and sum within each group
    archetype_col = pd.Series(
        {t: taxon_to_archetype.get(t, t) for t in df.index}, name="_archetype"
    )
    merged = df.copy()
    merged["_archetype"] = archetype_col
    new_otu_df = merged.groupby("_archetype", sort=False).sum()
    new_otu_df.index.name = None

    archetypes = list(new_otu_df.index)
    new_otu = OtuTable(new_otu_df, taxa_are_rows=True)

    new_tax = None
    if ps.tax_table is not None:
        tax_df = ps.tax_table.to_frame()
        keep_tax = [t for t in archetypes if t in tax_df.index]
        new_tax = TaxTable(tax_df.loc[keep_tax])

    new_tree = None
    if ps.phy_tree is not None:
        tree_tips = set(ps.phy_tree.tip_names)
        tree_keep = [t for t in archetypes if t in tree_tips]
        if len(tree_keep) >= 2:
            new_tree = ps.phy_tree.prune(tree_keep)

    return _rebuild_ps(
        ps, new_otu, tax=new_tax, tree=new_tree, refseq=_filter_refseq(ps, archetypes)
    )


# ---------------------------------------------------------------------------
# Ticket 3.8 — merge_phyloseq, merge_samples
# ---------------------------------------------------------------------------


def merge_phyloseq(*objs: Phyloseq) -> Phyloseq:
    """Merge two or more Phyloseq objects into one.

    OTU abundances are summed for overlapping (taxa, sample) pairs.  The union
    of all taxa and all samples is included (filling zeros where absent).

    Parameters
    ----------
    *objs:
        Two or more ``Phyloseq`` objects.

    Notes
    -----
    For ``sample_data``, ``tax_table``, and ``refseq``, overlapping keys are
    resolved first-wins (the earliest object in ``objs`` that defines the key).
    For the tree, the first non-null ``phy_tree`` is kept and pruned to the
    merged taxa by the constructor; differing trees across inputs are not
    reconciled.

    R reference: merge_phyloseq(...)
    """
    from pyloseq._phyloseq import Phyloseq as _Phyloseq

    if len(objs) < 2:
        raise ValueError("merge_phyloseq requires at least 2 Phyloseq objects.")

    all_otu_dfs = [_otu_taxa_rows(ps) for ps in objs]

    all_taxa = sorted(set().union(*[set(df.index) for df in all_otu_dfs]))
    all_samples = sorted(set().union(*[set(df.columns) for df in all_otu_dfs]))

    combined = pd.DataFrame(0.0, index=all_taxa, columns=all_samples)
    for df in all_otu_dfs:
        combined.loc[df.index, df.columns] += df

    new_otu = OtuTable(combined, taxa_are_rows=True)

    # Sample data: outer join, first-wins for duplicates
    sam_dfs = [ps.sample_data.to_frame() for ps in objs if ps.sample_data is not None]
    new_sam = None
    if sam_dfs:
        merged_sam = pd.concat(sam_dfs, axis=0)
        merged_sam = merged_sam.loc[~merged_sam.index.duplicated(keep="first")]
        merged_sam = merged_sam.reindex(pd.Index(all_samples))
        new_sam = SampleData(merged_sam.dropna(how="all"))

    # Tax table: outer join, first-wins for duplicates
    tax_dfs = [ps.tax_table.to_frame() for ps in objs if ps.tax_table is not None]
    new_tax = None
    if tax_dfs:
        merged_tax = pd.concat(tax_dfs, axis=0)
        merged_tax = merged_tax.loc[~merged_tax.index.duplicated(keep="first")]
        merged_tax = merged_tax.reindex(pd.Index(all_taxa))
        new_tax = TaxTable(merged_tax)

    # RefSeq: union across objects, first-wins for duplicate taxon IDs
    new_refseq = None
    objs_with_refseq = [ps for ps in objs if ps.refseq is not None]
    if objs_with_refseq:
        import skbio  # noqa: PLC0415

        merged_seqs: dict[str, skbio.DNA] = {}
        for ps in objs_with_refseq:
            rs = ps.refseq
            for name in rs.taxa_names:
                if name not in merged_seqs:  # first-wins
                    merged_seqs[name] = rs[name]
        if merged_seqs:
            new_refseq = RefSeq(merged_seqs)

    # Tree: take the first non-null tree; Phyloseq constructor prunes it
    new_tree = None
    for ps in objs:
        if ps.phy_tree is not None:
            new_tree = ps.phy_tree.copy()
            break

    return _Phyloseq(
        otu=new_otu,
        sam=new_sam,
        tax=new_tax,
        tree=new_tree,
        refseq=new_refseq,
        metadata={},
    )


def merge_samples(
    ps: Phyloseq,
    group_var: str,
    fn: Callable[[pd.Series], Any] | None = None,
) -> Phyloseq:
    """Collapse samples that share a value in a metadata variable.

    OTU abundances are **summed** within each group.  Sample metadata is
    aggregated: numeric columns via ``fn`` (default :func:`numpy.mean`), other
    columns via mode.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object (must have ``sample_data``).
    group_var:
        Column name in ``sample_data`` that defines the grouping.
    fn:
        Aggregation function for numeric metadata columns.

    Notes
    -----
    As in R's ``merge_samples``, *every* numeric metadata column is aggregated
    with ``fn`` (default mean). Numeric columns that are really identifiers
    (e.g. a numeric subject ID) will be averaged into meaningless values; drop
    or stringify such columns before merging if that is not desired.

    R reference: merge_samples(x, group, fun)
    """
    if ps.sample_data is None:
        raise pyloseqValidationError("merge_samples requires sample_data")

    if fn is None:
        fn = np.mean

    sam_df = ps.sample_data.to_frame()
    if group_var not in sam_df.columns:
        raise pyloseqValidationError(
            f"Variable '{group_var}' not found in sample_data. Available: {list(sam_df.columns)}"
        )

    groups: pd.Series = sam_df[group_var]
    otu_df = _otu_taxa_rows(ps)

    # Sum OTU abundances within each group
    new_otu_cols: dict[str, pd.Series] = {}
    for grp_name, grp_samples in groups.groupby(groups):
        new_otu_cols[str(grp_name)] = otu_df[grp_samples.index].sum(axis=1)
    new_otu_df = pd.DataFrame(new_otu_cols)

    # Aggregate sample metadata
    def _agg_col(col: pd.Series) -> Any:
        if pd.api.types.is_numeric_dtype(col):
            return fn(col)
        modes = col.mode()
        return modes.iloc[0] if len(modes) > 0 else col.iloc[0]

    new_sam_rows: dict[str, pd.Series] = {}
    for grp_name, grp_samples in groups.groupby(groups):
        grp_meta = sam_df.loc[grp_samples.index]
        new_sam_rows[str(grp_name)] = grp_meta.apply(_agg_col, axis=0)
    new_sam_df = pd.DataFrame(new_sam_rows).T
    new_sam_df.index.name = None

    new_otu = OtuTable(new_otu_df, taxa_are_rows=True)
    new_sam = SampleData(new_sam_df)

    return _rebuild_ps(ps, new_otu, sam=new_sam)


# ---------------------------------------------------------------------------
# Ticket 3.9 — psmelt
# ---------------------------------------------------------------------------


def psmelt(ps: Phyloseq) -> pd.DataFrame:
    """Melt a Phyloseq into a long-form tidy DataFrame.

    Returns one row per (OTU, Sample) pair with columns:
    ``["OTU", "Sample", "Abundance", *sample_variables, *rank_names]``.

    Parameters
    ----------
    ps:
        Source ``Phyloseq`` object.

    R reference: psmelt(physeq)
    """
    otu_df = _otu_taxa_rows(ps)

    # Stack to long form
    long = otu_df.stack().reset_index()
    long.columns = pd.Index(["OTU", "Sample", "Abundance"])

    if ps.sample_data is not None:
        sam_df = ps.sample_data.to_frame()
        long = long.merge(sam_df, left_on="Sample", right_index=True, how="left")

    if ps.tax_table is not None:
        tax_df = ps.tax_table.to_frame()
        long = long.merge(tax_df, left_on="OTU", right_index=True, how="left")

    return long.reset_index(drop=True)
