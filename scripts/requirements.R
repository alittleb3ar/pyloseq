#!/usr/bin/env Rscript
# Install R dependencies needed to run generate_golden.R.
# Pin versions to match PROVENANCE.json after first successful run.

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

BiocManager::install("phyloseq", ask = FALSE, update = FALSE)

pkgs <- c("vegan", "ape", "arrow")
for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

cat("All R dependencies installed.\n")
cat("Run: Rscript scripts/generate_golden.R\n")
