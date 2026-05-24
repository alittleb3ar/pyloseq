# phyloseq → pyloseq migration guide

Every public function in R phyloseq has a direct Python equivalent in pyloseq.
The table below lists R signatures alongside their pyloseq counterparts and any
behavioural differences worth noting.

---

## Core classes

| R class / constructor | pyloseq equivalent | Notes |
|---|---|---|
| `phyloseq(otu, sam, tax, tree, refseq)` | `Phyloseq(otu, sam, tax, tree, refseq)` | Same component names |
| `otu_table(matrix, taxa_are_rows)` | `OtuTable(df, taxa_are_rows)` | Accepts DataFrame or sparse matrix |
| `sample_data(data.frame)` | `SampleData(df)` | Index = sample names |
| `tax_table(matrix)` | `TaxTable(df)` | Index = taxa names; columns = ranks |
| `phy_tree(phylo)` | `PhyTree.from_newick(s)` | Wraps `skbio.TreeNode` |
| `refseq(DNAStringSet)` | `RefSeq({name: skbio.DNA(...)})` | Dict of `skbio.DNA` objects |

---

## Accessors

| R | pyloseq | Notes |
|---|---|---|
| `taxa_names(x)` | `ps.taxa_names` | `pd.Index` |
| `sample_names(x)` | `ps.sample_names` | `pd.Index` |
| `ntaxa(x)` | `ps.ntaxa` | `int` |
| `nsamples(x)` | `ps.nsamples` | `int` |
| `rank_names(x)` | `ps.rank_names` | `list[str]` |
| `sample_variables(x)` | `ps.sample_variables` | `list[str]` |
| `taxa_sums(x)` | `ps.taxa_sums()` | `pd.Series` |
| `sample_sums(x)` | `ps.sample_sums()` | `pd.Series` |
| `otu_table(x)` | `ps.otu_table` | `OtuTable` |
| `sample_data(x)` | `ps.sample_data` | `SampleData \| None` |
| `tax_table(x)` | `ps.tax_table` | `TaxTable \| None` |
| `phy_tree(x)` | `ps.phy_tree` | `PhyTree \| None` |
| `refseq(x)` | `ps.refseq` | `RefSeq \| None` |
| `get_variable(x, v)` | `ps.get_variable(v)` | `pd.Series` |
| `get_taxa(x, i)` | `ps.get_taxa(i)` | `pd.Series` |
| `get_sample(x, i)` | `ps.get_sample(i)` | `pd.Series` |

---

## I/O

| R | pyloseq | Notes |
|---|---|---|
| `import_biom(file)` | `read_biom(path)` | BIOM v1 and v2 |
| `export_biom(ps, file)` | `write_biom(ps, path)` | |
| `import_qiime(otufile, mapfile, ...)` | `read_qiime(otu_path, map_path)` | QIIME 1 OTU tables |
| `qza_to_phyloseq(qza)` | `read_qza(path)` | QIIME 2 artifacts |
| `import_mothur(list, count, tax)` | `read_mothur(list_path, ...)` | |
| *(no equivalent)* | `read_csv(otu_path, ...)` | CSV/TSV + optional sidecar files |
| *(no equivalent)* | `to_csv(ps, dir)` | Write each component as CSV |
| *(no equivalent)* | `ps.to_anndata()` | AnnData round-trip |
| *(no equivalent)* | `Phyloseq.from_anndata(ad)` | |

---

## Data manipulation

| R | pyloseq | Notes |
|---|---|---|
| `subset_samples(x, ...)` | `subset_samples(ps, predicate)` | `predicate(sample_df) → bool Series` |
| `subset_taxa(x, ...)` | `subset_taxa(ps, predicate)` | `predicate(tax_df) → bool Series` |
| `prune_samples(samples, x)` | `prune_samples(sample_names, ps)` | |
| `prune_taxa(taxa, x)` | `prune_taxa(taxa_names, ps)` | |
| `filter_taxa(x, fxn, prune)` | `filter_taxa(ps, predicate)` | Always prunes; returns new object |
| `kOverA(k, A)` | `kOverA(k, A)` | Returns predicate for `filter_taxa` |
| `transform_sample_counts(x, f)` | `transform_sample_counts(ps, f)` | `f(abundance_vector) → vector` |
| `rarefy_even_depth(x, rngseed)` | `rarefy_even_depth(ps, rng_seed)` | Uses `np.random.default_rng` |
| `tax_glom(x, taxrank)` | `tax_glom(ps, rank)` | Agglomerates at given rank |
| `tip_glom(x, h)` | `tip_glom(ps, h)` | Clusters tips by patristic distance |
| `merge_taxa(x, eqtaxa)` | `merge_taxa(ps, taxa)` | |
| `merge_samples(x, group)` | `merge_samples(ps, group_var)` | |
| `merge_phyloseq(...)` | `merge_phyloseq(*ps_objects)` | |
| `psmelt(x)` | `psmelt(ps)` or `ps.melt()` | Returns tidy `pd.DataFrame` |

---

## Analysis

| R | pyloseq | Notes |
|---|---|---|
| `estimate_richness(x, measures)` | `estimate_richness(ps, measures)` | Returns `pd.DataFrame` |
| `distance(x, method, type)` | `distance(ps, method, type)` or `ps.distance(method)` | Returns `skbio.DistanceMatrix` |
| `UniFrac(x, weighted, normalized)` | `unifrac(ps, weighted, normalized)` | Returns `skbio.DistanceMatrix` |
| `distanceMethodList` | `distance_method_list()` | Returns `dict[str, list[str]]` |
| `ordinate(x, method, distance, formula)` | `ordinate(ps, method, distance, formula)` or `ps.ordinate(...)` | Returns `skbio.OrdinationResults` |

---

## Plotting

| R | pyloseq | Notes |
|---|---|---|
| `plot_bar(x, x, fill, ...)` | `plot_bar(ps, x, fill, ...)` | Returns `plotnine.ggplot` |
| `plot_richness(x, x, measures, ...)` | `plot_richness(ps, x, measures, ...)` | |
| `plot_ordination(x, ord, type, color, ...)` | `plot_ordination(ps, ord, type, color, ...)` | |
| `plot_heatmap(x, ...)` | `plot_heatmap(ps, ...)` | |
| `plot_network(x, ...)` | `plot_network(ps, ...)` | Requires `networkx` |
| `make_network(x, ...)` | `make_network(ps, ...)` | Returns `networkx.Graph` |

---

## Key behavioural differences

| Topic | R phyloseq | pyloseq |
|---|---|---|
| **Mutation** | In-place via S4 replace methods | All functions return new objects |
| **OTU table orientation** | `taxa_are_rows` attribute | Same; `taxa_are_rows=True` is default |
| **AnnData convention** | N/A | `X` is samples × taxa (transposed from `taxa_are_rows=True`) |
| **Shannon entropy** | Natural log (nats) | Natural log (nats) — matches R vegan |
| **Jaccard** | Binary (presence/absence) by default | Binary by default (`binarize=True`) |
| **Random seed** | `set.seed()` global | Per-call `rng_seed` parameter |
| **Parallel UniFrac** | `parallel=TRUE, n_jobs` | `n_jobs` parameter |
| **Formula syntax** | R formula `~ Var1 + Var2` | String `"~Var1 + Var2"` |
| **PERMANOVA** | `vegan::adonis2()` (separate package) | `skbio.stats.distance.permanova(ps.distance(...), ...)` |
