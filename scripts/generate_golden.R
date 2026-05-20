#!/usr/bin/env Rscript
# Generate golden reference files from R phyloseq for use in Python tests.
#
# Outputs written to tests/golden/<dataset>/<component>.{parquet,nwk,fasta}
# and analysis function outputs to tests/golden/<dataset>/<function>/<hash>.parquet.
#
# Run from the repo root:
#   Rscript scripts/generate_golden.R

suppressPackageStartupMessages({
  library(phyloseq)
  library(arrow)
  library(ape)
  library(vegan)
})

# ---- version checks --------------------------------------------------------

check_version <- function(pkg, min_ver) {
  ver <- packageVersion(pkg)
  if (ver < package_version(min_ver)) {
    stop(sprintf(
      "%s >= %s required, found %s. Run: Rscript scripts/requirements.R",
      pkg, min_ver, ver
    ))
  }
  invisible(ver)
}

check_version("phyloseq", "1.44.0")
check_version("arrow",    "12.0.0")
check_version("vegan",    "2.6.0")
check_version("ape",      "5.7.0")

# ---- helpers ---------------------------------------------------------------

GOLDEN_DIR <- file.path("tests", "golden")
dir.create(GOLDEN_DIR, recursive = TRUE, showWarnings = FALSE)

#' Write a matrix or data.frame to Parquet via arrow.
write_parquet_safe <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  df <- as.data.frame(x)
  # Preserve rownames as a column called __index__
  df[["__index__"]] <- rownames(df)
  arrow::write_parquet(df, path)
  message("  wrote ", path)
}

#' Write a Series (named numeric vector) to Parquet.
write_series <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  df <- data.frame(
    `__index__` = names(x),
    value        = as.numeric(x),
    check.names  = FALSE
  )
  arrow::write_parquet(df, path)
  message("  wrote ", path)
}

#' Hash a named params list to 8-char hex.
#' Empty params → "default" (matches Python's load_golden() convention).
#' Non-empty params → first 8 chars of MD5 over the JSON representation.
params_hash <- function(params = list()) {
  if (length(params) == 0) return("default")
  json_str <- jsonlite::toJSON(params, auto_unbox = TRUE, digits = NA)
  substr(digest::digest(json_str, algo = "md5", serialize = FALSE), 1, 8)
}

# ---- per-dataset export ----------------------------------------------------

export_dataset <- function(name, ps) {
  base <- file.path(GOLDEN_DIR, name)
  dir.create(base, recursive = TRUE, showWarnings = FALSE)
  message("\n=== ", name, " ===")

  # Normalize all names: replace spaces with underscores so they match
  # what ape::write.tree produces in Newick (ape converts spaces → underscores).
  taxa_names(ps) <- gsub(" ", "_", taxa_names(ps))
  if (!is.null(sample_data(ps, errorIfNULL = FALSE))) {
    sample_names(ps) <- gsub(" ", "_", sample_names(ps))
  }

  # OTU table — taxa as rows, oriented as R stores it
  otu <- as(otu_table(ps), "matrix")
  write_parquet_safe(otu, file.path(base, "otu_table.parquet"))

  # Sample data
  if (!is.null(sample_data(ps, errorIfNULL = FALSE))) {
    sam <- as(sample_data(ps), "data.frame")
    write_parquet_safe(sam, file.path(base, "sample_data.parquet"))
  }

  # Tax table
  if (!is.null(tax_table(ps, errorIfNULL = FALSE))) {
    tax <- as(tax_table(ps), "matrix")
    write_parquet_safe(tax, file.path(base, "tax_table.parquet"))
  }

  # Tree — Newick
  if (!is.null(phy_tree(ps, errorIfNULL = FALSE))) {
    nwk_path <- file.path(base, "phy_tree.nwk")
    ape::write.tree(phy_tree(ps), file = nwk_path)
    message("  wrote ", nwk_path)
  }

  # Refseq — FASTA
  if (!is.null(refseq(ps, errorIfNULL = FALSE))) {
    fa_path <- file.path(base, "refseq.fasta")
    Biostrings::writeXStringSet(refseq(ps), filepath = fa_path)
    message("  wrote ", fa_path)
  }

  # ---- analysis function outputs ------------------------------------------

  # taxa_sums / sample_sums
  write_series(taxa_sums(ps),   file.path(base, "taxa_sums.parquet"))
  write_series(sample_sums(ps), file.path(base, "sample_sums.parquet"))

  # estimate_richness (alpha diversity)
  # Only valid on unrarefied integer counts; skip if data are non-integer
  if (all(otu_table(ps) == floor(otu_table(ps)))) {
    rich <- estimate_richness(ps)
    rich[["__index__"]] <- rownames(rich)
    dir.create(file.path(base, "estimate_richness"), recursive = TRUE, showWarnings = FALSE)
    # Use "default.parquet" for the no-params case so Python's load_golden() can find it
    # without a cross-language hash computation.
    arrow::write_parquet(rich, file.path(base, "estimate_richness", "default.parquet"))
    message("  wrote estimate_richness golden")
  }

  invisible(NULL)
}

# ---- load datasets & export ------------------------------------------------

# GlobalPatterns
data("GlobalPatterns")
export_dataset("GlobalPatterns", GlobalPatterns)

# enterotype
data("enterotype")
export_dataset("enterotype", enterotype)

# esophagus
data("esophagus")
export_dataset("esophagus", esophagus)

# soilrep
data("soilrep")
export_dataset("soilrep", soilrep)

# ---- Phase 3 manipulation golden files ------------------------------------

message("\n=== Phase 3 manipulation goldens ===")

# Re-load and re-normalize datasets for manipulation tests
data("GlobalPatterns"); data("enterotype"); data("esophagus")
taxa_names(GlobalPatterns) <- gsub(" ", "_", taxa_names(GlobalPatterns))
sample_names(GlobalPatterns) <- gsub(" ", "_", sample_names(GlobalPatterns))
taxa_names(enterotype) <- gsub(" ", "_", taxa_names(enterotype))
taxa_names(esophagus) <- gsub(" ", "_", taxa_names(esophagus))

# -- subset_samples --
gp_soil <- subset_samples(GlobalPatterns, SampleType == "Soil")
gp_soil_dir <- file.path(GOLDEN_DIR, "GlobalPatterns", "subset_samples_soil")
dir.create(gp_soil_dir, recursive = TRUE, showWarnings = FALSE)
write_parquet_safe(
  as(otu_table(gp_soil), "matrix"),
  file.path(gp_soil_dir, "otu_table.parquet")
)
write_parquet_safe(
  as(sample_data(gp_soil), "data.frame"),
  file.path(gp_soil_dir, "sample_data.parquet")
)
message("  subset_samples(GP, Soil): ", nsamples(gp_soil), " samples (expect 3)")

# -- subset_taxa --
gp_chlam <- subset_taxa(GlobalPatterns, Phylum == "Chlamydiae")
gp_chlam_dir <- file.path(GOLDEN_DIR, "GlobalPatterns", "subset_taxa_chlamydiae")
dir.create(gp_chlam_dir, recursive = TRUE, showWarnings = FALSE)
write_parquet_safe(
  as(otu_table(gp_chlam), "matrix"),
  file.path(gp_chlam_dir, "otu_table.parquet")
)
write_parquet_safe(
  as(tax_table(gp_chlam), "matrix"),
  file.path(gp_chlam_dir, "tax_table.parquet")
)
message("  subset_taxa(GP, Chlamydiae): ", ntaxa(gp_chlam), " taxa")

# -- filter_taxa (kOverA equivalent) --
# filter_taxa(enterotype, kOverA(5, 2e-5), prune=TRUE) → 416 taxa x 280 samples
et_filtered <- filter_taxa(enterotype, function(x) sum(x > 2e-5) >= 5, TRUE)
et_filt_dir <- file.path(GOLDEN_DIR, "enterotype", "filter_taxa_kOverA_5_2e-5")
dir.create(et_filt_dir, recursive = TRUE, showWarnings = FALSE)
write_parquet_safe(
  as(otu_table(et_filtered), "matrix"),
  file.path(et_filt_dir, "otu_table.parquet")
)
message("  filter_taxa(enterotype, kOverA(5,2e-5)): ", ntaxa(et_filtered),
        " taxa x ", nsamples(et_filtered), " samples (expect 416 x 280)")

# -- tax_glom at Family --
gp_fam <- tax_glom(GlobalPatterns, "Family")
gp_fam_dir <- file.path(GOLDEN_DIR, "GlobalPatterns", "tax_glom_Family")
dir.create(gp_fam_dir, recursive = TRUE, showWarnings = FALSE)
write_parquet_safe(
  as(otu_table(gp_fam), "matrix"),
  file.path(gp_fam_dir, "otu_table.parquet")
)
write_parquet_safe(
  as(tax_table(gp_fam), "matrix"),
  file.path(gp_fam_dir, "tax_table.parquet")
)
write_series(taxa_sums(gp_fam), file.path(gp_fam_dir, "taxa_sums.parquet"))
message("  tax_glom(GP, Family): ", ntaxa(gp_fam), " taxa")

# -- tax_glom at Genus --
gp_gen <- tax_glom(GlobalPatterns, "Genus")
gp_gen_dir <- file.path(GOLDEN_DIR, "GlobalPatterns", "tax_glom_Genus")
dir.create(gp_gen_dir, recursive = TRUE, showWarnings = FALSE)
write_parquet_safe(
  as(otu_table(gp_gen), "matrix"),
  file.path(gp_gen_dir, "otu_table.parquet")
)
write_series(taxa_sums(gp_gen), file.path(gp_gen_dir, "taxa_sums.parquet"))
message("  tax_glom(GP, Genus): ", ntaxa(gp_gen), " taxa")

# -- merge_samples by SampleType --
gp_merged <- merge_samples(GlobalPatterns, "SampleType")
gp_ms_dir <- file.path(GOLDEN_DIR, "GlobalPatterns", "merge_samples_SampleType")
dir.create(gp_ms_dir, recursive = TRUE, showWarnings = FALSE)
write_parquet_safe(
  as(otu_table(gp_merged), "matrix"),
  file.path(gp_ms_dir, "otu_table.parquet")
)
write_series(sample_sums(gp_merged), file.path(gp_ms_dir, "sample_sums.parquet"))
message("  merge_samples(GP, SampleType): ", nsamples(gp_merged),
        " samples (expect 9)")

# ---- UniFrac golden files (esophagus — small, fast) ------------------------

message("\n=== UniFrac goldens (esophagus) ===")

uf_un <- UniFrac(esophagus, weighted = FALSE, normalized = TRUE)
uf_wt <- UniFrac(esophagus, weighted = TRUE,  normalized = TRUE)

uf_to_df <- function(dm) {
  m <- as.matrix(dm)
  df <- as.data.frame(m)
  df[["__index__"]] <- rownames(m)
  df
}

dir.create(file.path(GOLDEN_DIR, "esophagus", "unifrac_unweighted"), recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(
  uf_to_df(uf_un),
  file.path(GOLDEN_DIR, "esophagus", "unifrac_unweighted", "normalized.parquet")
)
message("  wrote unifrac_unweighted golden")

dir.create(file.path(GOLDEN_DIR, "esophagus", "unifrac_weighted"), recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(
  uf_to_df(uf_wt),
  file.path(GOLDEN_DIR, "esophagus", "unifrac_weighted", "normalized.parquet")
)
message("  wrote unifrac_weighted golden")

# ---- PROVENANCE.json -------------------------------------------------------

provenance <- list(
  phyloseq    = as.character(packageVersion("phyloseq")),
  vegan       = as.character(packageVersion("vegan")),
  ape         = as.character(packageVersion("ape")),
  arrow       = as.character(packageVersion("arrow")),
  R           = R.version.string,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
)

jsonlite::write_json(
  provenance,
  file.path(GOLDEN_DIR, "PROVENANCE.json"),
  pretty   = TRUE,
  auto_unbox = TRUE
)
message("\nwrote tests/golden/PROVENANCE.json")
message("\nDone. Commit tests/golden/ via Git LFS.")
