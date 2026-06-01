"""Core Phyloseq container

R reference: phyloseq::phyloseq-class
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

from pyloseq._distances import distance as _distance
from pyloseq._exceptions import pyloseqValidationError
from pyloseq._otu_table import OtuTable
from pyloseq._refseq import RefSeq
from pyloseq._sample_data import SampleData
from pyloseq._tax_table import TaxTable
from pyloseq._tree import PhyTree

if TYPE_CHECKING:
    from skbio.stats.distance import DistanceMatrix
    from skbio.stats.ordination import OrdinationResults


class Phyloseq:
    """Container for microbiome data: OTU table + optional metadata components.

    Mirrors R's ``phyloseq-class``. The constructor accepts any subset of
    components, runs the validator suite, and silently prunes to the
    intersection of names across components (unless ``strict=True``).

    By default, pruning during construction emits a warning so the data loss is
    discoverable; pass ``quiet=True`` to suppress it.

    R reference: phyloseq::phyloseq(otu_table, sample_data, tax_table, phy_tree, refseq)
    """

    def __init__(
        self,
        otu: OtuTable,
        sam: SampleData | None = None,
        tax: TaxTable | None = None,
        tree: PhyTree | None = None,
        refseq: RefSeq | None = None,
        strict: bool = False,
        quiet: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._otu = otu
        self._sam = sam
        self._tax = tax
        self._tree = tree
        self._refseq = refseq
        self.metadata: dict[str, Any] = metadata or {}
        _validate(
            self,
            strict=strict,
            warn_on_prune=not quiet,
            context="Construction",
        )

    # ------------------------------------------------------------------
    # Component properties (with setters that re-validate)
    # ------------------------------------------------------------------

    def _revalidate(self) -> None:
        """Re-run the validator after a component is replaced via a setter."""
        _validate(
            self,
            strict=False,
            warn_on_prune=True,
            context="Component assignment",
        )

    @property
    def otu_table(self) -> OtuTable:
        """The OTU/feature abundance table.

        R reference: otu_table(x)
        """
        return self._otu

    @otu_table.setter
    def otu_table(self, value: OtuTable) -> None:
        self._otu = value
        self._revalidate()

    @property
    def sample_data(self) -> SampleData | None:
        """Per-sample metadata, or ``None`` if not provided.

        R reference: sample_data(x)
        """
        return self._sam

    @sample_data.setter
    def sample_data(self, value: SampleData | None) -> None:
        self._sam = value
        self._revalidate()

    @property
    def tax_table(self) -> TaxTable | None:
        """Taxonomic classification table, or ``None`` if not provided.

        R reference: tax_table(x)
        """
        return self._tax

    @tax_table.setter
    def tax_table(self, value: TaxTable | None) -> None:
        self._tax = value
        self._revalidate()

    @property
    def phy_tree(self) -> PhyTree | None:
        """Phylogenetic tree, or ``None`` if not provided.

        R reference: phy_tree(x)
        """
        return self._tree

    @phy_tree.setter
    def phy_tree(self, value: PhyTree | None) -> None:
        self._tree = value
        self._revalidate()

    @property
    def refseq(self) -> RefSeq | None:
        """Reference sequences, or ``None`` if not provided.

        R reference: refseq(x)
        """
        return self._refseq

    @refseq.setter
    def refseq(self, value: RefSeq | None) -> None:
        self._refseq = value
        self._revalidate()

    # ------------------------------------------------------------------
    # Accessors — mirror R phyloseq function-based API
    # ------------------------------------------------------------------

    @property
    def taxa_names(self) -> pd.Index:
        """Taxon identifiers from the OTU table.

        R reference: taxa_names(x)
        """
        return self._otu.taxa_names

    @property
    def sample_names(self) -> pd.Index:
        """Sample identifiers from the OTU table.

        R reference: sample_names(x)
        """
        return self._otu.sample_names

    @property
    def ntaxa(self) -> int:
        """Number of taxa.

        R reference: ntaxa(x)
        """
        return self._otu.ntaxa

    @property
    def nsamples(self) -> int:
        """Number of samples.

        R reference: nsamples(x)
        """
        return self._otu.nsamples

    @property
    def sample_variables(self) -> list[str]:
        """Names of sample metadata columns, or ``[]`` if no sample data.

        R reference: sample_variables(x)
        """
        if self._sam is None:
            return []
        return list(self._sam.variables)

    @property
    def rank_names(self) -> list[str]:
        """Taxonomic rank names, or ``[]`` if no tax table.

        R reference: rank_names(x)
        """
        if self._tax is None:
            return []
        return self._tax.rank_names

    def get_variable(self, v: str) -> pd.Series:
        """Return a sample metadata column as a Series.

        R reference: get_variable(x, v)
        """
        if self._sam is None:
            raise ValueError("No sample_data attached to this Phyloseq object")
        return self._sam.to_frame()[v]

    def get_taxa(self, i: str) -> pd.Series:
        """Return the abundance vector for a single taxon across all samples.

        R reference: get_taxa(x, i)
        """
        df = self._otu.to_dataframe()
        if self._otu.taxa_are_rows:
            return df.loc[i]
        return df[i]

    def get_sample(self, i: str) -> pd.Series:
        """Return the abundance vector for a single sample across all taxa.

        R reference: get_sample(x, i)
        """
        df = self._otu.to_dataframe()
        if self._otu.taxa_are_rows:
            return df[i]
        return df.loc[i]

    def taxa_sums(self) -> pd.Series:
        """Sum of abundances across all samples for each taxon.

        R reference: taxa_sums(x)
        """
        return self._otu.taxa_sums()

    def sample_sums(self) -> pd.Series:
        """Sum of abundances across all taxa for each sample.

        R reference: sample_sums(x)
        """
        return self._otu.sample_sums()

    def melt(self) -> pd.DataFrame:
        """Melt to a long-form tidy DataFrame (one row per OTU × Sample pair).

        Equivalent to the free function :func:`pyloseq.psmelt`.

        R reference: psmelt(physeq)
        """
        from pyloseq._manipulation import psmelt

        return psmelt(self)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"Phyloseq({self.ntaxa} taxa × {self.nsamples} samples)"]
        if self._sam is not None:
            parts.append(f"  sample_data: {len(self._sam.variables)} variables")
        if self._tax is not None:
            parts.append(f"  tax_table:   {len(self._tax.rank_names)} ranks")
        if self._tree is not None:
            parts.append(f"  phy_tree:    {self._tree.n_tips} tips")
        if self._refseq is not None:
            parts.append(f"  refseq:      {len(self._refseq)} sequences")
        return "\n".join(parts)

    def distance(self, method: str = "bray", **kwargs: Any) -> DistanceMatrix:
        """Compute a pairwise distance matrix.

        Thin wrapper around :func:`pyloseq.distance` — returns a
        ``skbio.stats.distance.DistanceMatrix`` usable directly with
        ``skbio.stats.distance.permanova`` / ``anosim``.

        Parameters
        ----------
        method:
            Distance method string (e.g. ``"bray"``, ``"unifrac"``).
        **kwargs:
            Forwarded to the underlying implementation.

        R reference: distance(physeq, method)
        """

        return _distance(self, method, **kwargs)

    def ordinate(
        self,
        method: str = "PCoA",
        distance: str = "bray",
        formula: str | None = None,
        **kwargs: Any,
    ) -> OrdinationResults:
        """Run multivariate ordination.

        Thin wrapper around :func:`pyloseq.ordinate` — returns an
        ``skbio.stats.ordination.OrdinationResults``.

        Parameters
        ----------
        method:
            Ordination method: ``"PCoA"``, ``"NMDS"``, ``"CCA"``, etc.
        distance:
            Distance method or pre-computed ``DistanceMatrix``.
        formula:
            Model formula for constrained methods (e.g. ``"~SampleType"``).
        **kwargs:
            Forwarded to the underlying implementation.

        R reference: ordinate(physeq, method, distance, formula)
        """
        from pyloseq._ordination import ordinate as _ordinate

        return _ordinate(
            self, method=method, distance=distance, formula=formula, **kwargs
        )


# ------------------------------------------------------------------
# Validator suite (Ticket 1.5)
# ------------------------------------------------------------------


def _emit_prune_warning(n_drop: int, unit: str, context: str) -> None:
    """Emit a consistent, context-aware pruning warning."""
    warnings.warn(
        f"{context} pruned {n_drop} {unit} to the shared intersection.",
        stacklevel=3,
    )


def _prune_tree(tree: PhyTree, keep: list[str]) -> PhyTree | None:
    """Prune ``tree`` to the ``keep`` tips, warning if the result is unusable.

    Returns the pruned tree, the original tree unchanged if no tips need
    dropping, or ``None`` (with a warning) if fewer than two tips survive — a
    tree needs at least two tips to be meaningful.
    """
    tip_set = set(tree.tip_names)
    tree_keep = [t for t in keep if t in tip_set]
    if len(tree_keep) == len(tip_set):
        return tree  # nothing to prune
    if len(tree_keep) < 2:
        warnings.warn(
            f"phy_tree dropped: intersection left {len(tree_keep)} tip(s), "
            "need >= 2 for a meaningful tree.",
            stacklevel=3,
        )
        return None
    return tree.prune(tree_keep)


def _validate(
    ps: Phyloseq,
    strict: bool,
    warn_on_prune: bool = False,
    context: str = "Validation",
) -> None:
    """Run the full component-consistency validator suite.

    Checks (in order):
    1. OtuTable is present
    2. Taxa-name intersection across otu_table, tax_table, and refseq is non-empty
       (phy_tree is intentionally excluded — it is pruned TO the OTU table, not vice versa)
    3. Sample-name intersection between OtuTable and SampleData is non-empty
    4. Prune all components to the intersection (or raise in strict mode)

    R reference: validObject(phyloseq(...))
    """
    # Rule 1 — OtuTable required
    if ps._otu is None:
        raise pyloseqValidationError("otu_table is required")

    otu_taxa = set(ps._otu.taxa_names)

    # Rule 2 — taxa-name intersection across otu_table, tax_table, refseq
    # phy_tree is NOT included: a tree with extra/fewer tips is pruned to match the OTU table
    components_taxa: list[set[str]] = [otu_taxa]
    if ps._tax is not None:
        components_taxa.append(set(ps._tax.taxa_names))
    if ps._refseq is not None:
        components_taxa.append(set(ps._refseq.taxa_names))

    taxa_intersection: set[str] = components_taxa[0]
    for s in components_taxa[1:]:
        taxa_intersection = taxa_intersection & s

    if len(taxa_intersection) == 0 and len(otu_taxa) > 0 and len(components_taxa) > 1:
        raise pyloseqValidationError(
            "Component taxa/OTU names do not match. Try taxa_names()"
        )

    # Rules 3 & 4 — prune to intersection (or raise in strict mode)
    # Pruning may affect the OTU table (taxa_intersection < otu_taxa) OR other
    # components (e.g. tax_table has taxa not in the new smaller OTU table).
    max_component_size = max(len(s) for s in components_taxa)
    any_pruning_needed = len(taxa_intersection) < max_component_size
    if any_pruning_needed:
        if strict:
            only_otu = otu_taxa - taxa_intersection
            raise pyloseqValidationError(
                f"Component taxa/OTU names do not match. "
                f"{len(only_otu)} taxa present only in otu_table. "
                "Try taxa_names()"
            )
        if warn_on_prune:
            n_drop = max_component_size - len(taxa_intersection)
            _emit_prune_warning(n_drop, "taxa", context)
        _prune_to_taxa(ps, sorted(taxa_intersection))
    elif ps._tree is not None:
        # Even when no OTU/tax/refseq pruning is needed, the tree may have
        # extra tips that are not in the OTU table — prune it unconditionally.
        ps._tree = _prune_tree(ps._tree, sorted(taxa_intersection))

    if ps._sam is not None:
        otu_samples = set(ps._otu.sample_names)
        sam_samples = set(ps._sam.sample_names)
        sample_intersection = otu_samples & sam_samples
        if len(sample_intersection) == 0:
            raise pyloseqValidationError(
                "Component sample names do not match. Try sample_names()"
            )
        if len(sample_intersection) < len(otu_samples):
            if strict:
                raise pyloseqValidationError(
                    "Component sample names do not match. Try sample_names()"
                )
            if warn_on_prune:
                n_drop = len(otu_samples) - len(sample_intersection)
                _emit_prune_warning(n_drop, "samples", context)
            _prune_to_samples(ps, sorted(sample_intersection))


def _prune_to_taxa(ps: Phyloseq, keep: list[str]) -> None:
    """Mutate ps in-place, reducing all components to ``keep`` taxa."""
    keep_idx = pd.Index(keep)

    # OtuTable
    df = ps._otu.to_dataframe()
    if ps._otu.taxa_are_rows:
        ps._otu = OtuTable(df.loc[keep_idx], taxa_are_rows=True)
    else:
        ps._otu = OtuTable(df[keep_idx], taxa_are_rows=False)

    # TaxTable
    if ps._tax is not None:
        tax_df = ps._tax.to_frame()
        ps._tax = TaxTable(tax_df.loc[keep_idx])

    # PhyTree — prune tips not in keep (shared helper, warns if < 2 survive)
    if ps._tree is not None:
        ps._tree = _prune_tree(ps._tree, keep)

    # RefSeq
    if ps._refseq is not None:
        import skbio

        ref_names = set(ps._refseq.taxa_names)
        new_seqs: dict[str, skbio.DNA] = {
            k: ps._refseq[k] for k in keep if k in ref_names
        }
        ps._refseq = RefSeq(new_seqs)


def _prune_to_samples(ps: Phyloseq, keep: list[str]) -> None:
    """Mutate ps in-place, reducing OtuTable and SampleData to ``keep`` samples."""
    keep_idx = pd.Index(keep)

    df = ps._otu.to_dataframe()
    if ps._otu.taxa_are_rows:
        ps._otu = OtuTable(df[keep_idx], taxa_are_rows=True)
    else:
        ps._otu = OtuTable(df.loc[keep_idx], taxa_are_rows=False)

    if ps._sam is not None:
        sam_df = ps._sam.to_frame()
        ps._sam = SampleData(sam_df.loc[keep_idx])
