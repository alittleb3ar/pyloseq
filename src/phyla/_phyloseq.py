"""Core Phyloseq container and component-consistency validator.

R reference: phyloseq::phyloseq-class
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pyloseq._exceptions import pyloseqValidationError
from pyloseq._otu_table import OtuTable
from pyloseq._refseq import RefSeq
from pyloseq._sample_data import SampleData
from pyloseq._tax_table import TaxTable
from pyloseq._tree import PhyTree


class Phyloseq:
    """Container for microbiome data: OTU table + optional metadata components.

    Mirrors R's ``phyloseq-class``. The constructor accepts any subset of
    components, runs the validator suite, and silently prunes to the
    intersection of names across components (unless ``strict=True``).

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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._otu = otu
        self._sam = sam
        self._tax = tax
        self._tree = tree
        self._refseq = refseq
        self.metadata: dict[str, Any] = metadata or {}
        _validate(self, strict=strict)

    # ------------------------------------------------------------------
    # Component properties (with setters that re-validate)
    # ------------------------------------------------------------------

    @property
    def otu_table(self) -> OtuTable:
        """The OTU/feature abundance table.

        R reference: otu_table(x)
        """
        return self._otu

    @otu_table.setter
    def otu_table(self, value: OtuTable) -> None:
        self._otu = value
        _validate(self, strict=False)

    @property
    def sample_data(self) -> SampleData | None:
        """Per-sample metadata, or ``None`` if not provided.

        R reference: sample_data(x)
        """
        return self._sam

    @sample_data.setter
    def sample_data(self, value: SampleData | None) -> None:
        self._sam = value
        _validate(self, strict=False)

    @property
    def tax_table(self) -> TaxTable | None:
        """Taxonomic classification table, or ``None`` if not provided.

        R reference: tax_table(x)
        """
        return self._tax

    @tax_table.setter
    def tax_table(self, value: TaxTable | None) -> None:
        self._tax = value
        _validate(self, strict=False)

    @property
    def phy_tree(self) -> PhyTree | None:
        """Phylogenetic tree, or ``None`` if not provided.

        R reference: phy_tree(x)
        """
        return self._tree

    @phy_tree.setter
    def phy_tree(self, value: PhyTree | None) -> None:
        self._tree = value
        _validate(self, strict=False)

    @property
    def refseq(self) -> RefSeq | None:
        """Reference sequences, or ``None`` if not provided.

        R reference: refseq(x)
        """
        return self._refseq

    @refseq.setter
    def refseq(self, value: RefSeq | None) -> None:
        self._refseq = value
        _validate(self, strict=False)

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
    # Convenience wrappers for analysis functions (Ticket 5.2)
    # ------------------------------------------------------------------

    def distance(self, method: str = "bray", **kwargs: Any) -> Any:
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
        from pyloseq._distances import distance as _distance

        return _distance(self, method, **kwargs)

    def ordinate(
        self,
        method: str = "PCoA",
        distance: str = "bray",
        formula: str | None = None,
        **kwargs: Any,
    ) -> Any:
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

        return _ordinate(self, method=method, distance=distance, formula=formula, **kwargs)

    # ------------------------------------------------------------------
    # AnnData interop (Ticket 5.1)
    # ------------------------------------------------------------------

    def to_anndata(self) -> Any:
        """Convert to an ``anndata.AnnData`` object.

        Layout (matching AnnData convention):
        - ``X``   — ``float`` abundance matrix, shape ``(n_samples, n_taxa)``
        - ``obs`` — sample metadata (``sample_data``)
        - ``var`` — taxonomic ranks (``tax_table``)
        - ``uns["phy_tree"]``  — Newick string (if tree present)
        - ``uns["refseq"]``    — ``{taxon: sequence_str}`` dict (if refseq present)

        Use :meth:`from_anndata` to recover the original ``Phyloseq``.

        .. note::
            AnnData stores samples as rows.  The OTU table is therefore
            **transposed** relative to the taxa-are-rows convention common in
            phyloseq. :meth:`from_anndata` handles this automatically.
        """
        try:
            import anndata  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "anndata is required for to_anndata(). "
                "Install it with: pip install anndata"
            ) from exc

        # X: samples × taxa (AnnData convention)
        otu_df = self._otu.to_dataframe()
        if self._otu.taxa_are_rows:
            otu_df = otu_df.T  # → samples × taxa

        obs = self._sam.to_frame() if self._sam is not None else pd.DataFrame(index=otu_df.index)
        var = self._tax.to_frame() if self._tax is not None else pd.DataFrame(index=otu_df.columns)

        # Align indices
        obs = obs.loc[otu_df.index] if len(obs) > 0 else obs.reindex(otu_df.index)
        var = var.loc[otu_df.columns] if len(var) > 0 else var.reindex(otu_df.columns)

        uns: dict[str, Any] = {}
        if self._tree is not None:
            uns["phy_tree"] = self._tree.to_newick()
        if self._refseq is not None:
            uns["refseq"] = {name: str(self._refseq[name]) for name in self._refseq.names}

        return anndata.AnnData(
            X=otu_df.values.astype(float),
            obs=obs,
            var=var,
            uns=uns,
        )

    @classmethod
    def from_anndata(cls, ad: Any) -> Phyloseq:
        """Reconstruct a ``Phyloseq`` from an ``anndata.AnnData`` object.

        Assumes the AnnData was produced by :meth:`to_anndata` (or follows the
        same layout: samples as rows in ``X``, optional ``uns["phy_tree"]`` /
        ``uns["refseq"]``).

        Parameters
        ----------
        ad:
            ``AnnData`` object.

        Returns
        -------
        Phyloseq
        """
        # X is samples × taxa — store transposed (taxa × samples, taxa_are_rows=True)
        otu_df = pd.DataFrame(
            ad.X if not hasattr(ad.X, "toarray") else ad.X.toarray(),
            index=ad.obs_names,
            columns=ad.var_names,
        ).T
        otu = OtuTable(otu_df, taxa_are_rows=True)

        sam = SampleData(pd.DataFrame(ad.obs)) if ad.obs.shape[1] > 0 else None
        tax = TaxTable(pd.DataFrame(ad.var)) if ad.var.shape[1] > 0 else None

        tree: PhyTree | None = None
        if "phy_tree" in ad.uns and ad.uns["phy_tree"]:
            tree = PhyTree.from_newick(ad.uns["phy_tree"])

        refseq: RefSeq | None = None
        if "refseq" in ad.uns and ad.uns["refseq"]:
            import skbio  # type: ignore[import]
            refseq = RefSeq({k: skbio.DNA(v) for k, v in ad.uns["refseq"].items()})

        return cls(otu=otu, sam=sam, tax=tax, tree=tree, refseq=refseq)

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


# ------------------------------------------------------------------
# Validator suite (Ticket 1.5)
# ------------------------------------------------------------------

def _validate(ps: Phyloseq, strict: bool) -> None:
    """Run the full component-consistency validator suite.

    Checks (in order):
    1. OtuTable is present
    2. Taxa-name intersection across all present components is non-empty
    3. Sample-name intersection between OtuTable and SampleData is non-empty
    4. Tree tips are aligned with taxa names (prune or raise)
    5. RefSeq names are aligned with taxa names (prune or raise)

    R reference: validObject(phyloseq(...))
    """
    # Rule 1 — OtuTable required
    if ps._otu is None:
        raise pyloseqValidationError("otu_table is required")

    otu_taxa = set(ps._otu.taxa_names)

    # Rule 2 — taxa-name intersection across all components must be non-empty
    components_taxa: list[set[str]] = [otu_taxa]
    if ps._tax is not None:
        components_taxa.append(set(ps._tax.names))
    if ps._tree is not None:
        components_taxa.append(set(ps._tree.tip_names))
    if ps._refseq is not None:
        components_taxa.append(set(ps._refseq.names))

    taxa_intersection: set[str] = components_taxa[0]
    for s in components_taxa[1:]:
        taxa_intersection = taxa_intersection & s

    if len(taxa_intersection) == 0 and len(otu_taxa) > 0 and len(components_taxa) > 1:
        raise pyloseqValidationError(
            "Component taxa/OTU names do not match. Try taxa_names()"
        )

    # Rule 3 — sample-name intersection between OtuTable and SampleData
    if ps._sam is not None:
        otu_samples = set(ps._otu.sample_names)
        sam_samples = set(ps._sam.names)
        sample_intersection = otu_samples & sam_samples
        if len(sample_intersection) == 0:
            raise pyloseqValidationError(
                "Component sample names do not match. Try sample_names()"
            )

    # Rules 4 & 5 — prune to intersection (or raise in strict mode)
    if len(taxa_intersection) < len(otu_taxa):
        if strict:
            only_otu = otu_taxa - taxa_intersection
            raise pyloseqValidationError(
                f"Component taxa/OTU names do not match. "
                f"{len(only_otu)} taxa present only in otu_table. "
                "Try taxa_names()"
            )
        # Prune silently
        _prune_to_taxa(ps, sorted(taxa_intersection))

    if ps._sam is not None:
        otu_samples = set(ps._otu.sample_names)
        sam_samples = set(ps._sam.names)
        sample_intersection = otu_samples & sam_samples
        if len(sample_intersection) < len(otu_samples):
            if strict:
                raise pyloseqValidationError(
                    "Component sample names do not match. Try sample_names()"
                )
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

    # PhyTree — prune tips not in keep
    if ps._tree is not None:
        _tip_set = set(ps._tree.tip_names)
        tree_keep = [t for t in keep if t in _tip_set]
        if tree_keep:
            ps._tree = ps._tree.prune(tree_keep)
        else:
            ps._tree = None

    # RefSeq
    if ps._refseq is not None:
        import skbio
        new_seqs: dict[str, skbio.DNA] = {
            k: ps._refseq[k] for k in keep if k in ps._refseq.names
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
