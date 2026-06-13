# 1. Torondel et al. 2016 — Pit latrine bacterial ecology (environmental)

**Citation:** Torondel B, Ensink JHJ, Gundogdu O, Ijaz UZ, Parkhill J, Abdelahi F, Nguyen V-A, Sudgen S, Gibson W, Walker AW, Quince C. "Assessment of the influence of intrinsic environmental and geographical factors on the bacterial ecology of pit latrines." *Microbial Biotechnology*, 2016, 9(2):209–223. DOI: 10.1111/1751-7915.12334.

**Summary:** Bacterial diversity and composition was studied in 30 latrines in Tanzania and Vietnam using pyrosequencing of 16S rRNA genes, correlating community composition with environmental variables (pH, temperature, organic matter content/composition) and geography.

**How phyloseq is used:** This dataset is the basis of the widely-used **microbiomeSeq** tutorial built on phyloseq. Analyses include rarefaction curves, taxonomic composition bar plots (phylum and family), alpha diversity, beta diversity (PCoA/NMDS), differential abundance (DESeq2 via `phyloseq_to_deseq2`), and co-occurrence network analysis. A community GitHub mini-project (khairilradzali/metagenomicanalysis_mini_project) reproduces exactly these phyloseq analyses on this data.

**Where the data is:** Processed tables hosted at the University of Glasgow (Umer Ijaz's bioinformatics resources page, `userweb.eng.gla.ac.uk/umer.ijaz/bioinformatics/ecological.html`): an OTU abundance table (`All_Good_P2_C03.csv`) and an environmental/metadata table (`ENV_pitlatrine.csv`), plus a taxonomy file and tree. Resulting object: 8,883 taxa × 81 samples, 14 sample variables. **Format: CSV/text tables — confirmed processed, not raw FASTQ and not .rds.**

**Features exercised / gaps beyond 2013:** `make_network`/`plot_network` co-occurrence networks, `phyloseq_to_deseq2` → DESeq2 integration, NMDS, `tax_glom` agglomeration, rarefaction. Networks and DESeq2 integration go beyond McMurdie & Holmes 2013.

**Reproduction notes:** Very accessible; the microbiomeSeq tutorial and the GitHub mini-project provide reference code. Minor caveat: the OTU table must be transposed on import.
