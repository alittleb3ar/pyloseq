# Testing Utilities

Helpers for loading the pre-generated golden-file fixtures in tests. All loaders
return a `dict` with keys `otu_table`, `sample_data`, `tax_table`, and optionally
`phy_tree_newick` / `refseq_path` depending on the dataset.

::: pyloseq.testing.load_global_patterns_reference

---

::: pyloseq.testing.load_enterotype_reference

---

::: pyloseq.testing.load_esophagus_reference

---

::: pyloseq.testing.load_soilrep_reference

---

::: pyloseq.testing.load_golden
