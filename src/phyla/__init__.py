from importlib.metadata import version

from phyla._distances import distance, distance_method_list, unifrac
from phyla._diversity import estimate_richness
from phyla._exceptions import PhylaValidationError
from phyla._manipulation import (
    filter_taxa,
    kOverA,
    merge_phyloseq,
    merge_samples,
    merge_taxa,
    prune_samples,
    prune_taxa,
    psmelt,
    rarefy_even_depth,
    subset_samples,
    subset_taxa,
    tax_glom,
    tip_glom,
    transform_sample_counts,
)
from phyla._ordination import ordinate
from phyla._otu_table import OtuTable
from phyla._phyloseq import Phyloseq
from phyla._refseq import RefSeq
from phyla._sample_data import SampleData
from phyla._tax_table import TaxTable
from phyla._tree import PhyTree
from phyla.io import (
    read_biom,
    read_csv,
    read_mothur,
    read_qiime,
    read_qza,
    select_mothur_cutoff,
    show_mothur_cutoffs,
    to_csv,
    write_biom,
    write_qza,
)
from phyla.plotting import (
    make_network,
    plot_bar,
    plot_heatmap,
    plot_network,
    plot_ordination,
    plot_richness,
)

__version__ = version("phyla")

__all__ = [
    "__version__",
    # Core containers
    "Phyloseq",
    "OtuTable",
    "SampleData",
    "TaxTable",
    "PhyTree",
    "RefSeq",
    "PhylaValidationError",
    # I/O
    "read_biom",
    "write_biom",
    "read_qza",
    "write_qza",
    "read_qiime",
    "read_mothur",
    "show_mothur_cutoffs",
    "select_mothur_cutoff",
    "read_csv",
    "to_csv",
    # Data manipulation
    "subset_samples",
    "subset_taxa",
    "prune_samples",
    "prune_taxa",
    "filter_taxa",
    "kOverA",
    "transform_sample_counts",
    "rarefy_even_depth",
    "tax_glom",
    "tip_glom",
    "merge_phyloseq",
    "merge_samples",
    "merge_taxa",
    "psmelt",
    # Analysis
    "estimate_richness",
    "distance",
    "distance_method_list",
    "unifrac",
    "ordinate",
    # Plotting
    "plot_bar",
    "plot_richness",
    "plot_ordination",
    "plot_heatmap",
    "make_network",
    "plot_network",
]
