#!/usr/bin/env Rscript
# Install R dependencies needed to run generate_golden.R.
# Pin versions to match PROVENANCE.json after first successful run.

options(
  Ncpus = max(1L, parallel::detectCores()),
  timeout = 600
)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

BiocManager::install("phyloseq", ask = FALSE, update = FALSE)

for (pkg in c("vegan", "ape", "arrow")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

cat("All R dependencies installed.\n")
cat("Run: Rscript scripts/generate_golden.R\n")
