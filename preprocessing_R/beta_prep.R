# install.packages("BiocManager")
# BiocManager::install(c(
#   "minfi",
#   "IlluminaHumanMethylation450kanno.ilmn12.hg19",
#   "GenomicRanges",
#   "data.table"
# ))
# 
# BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")


# library(data.table)
# beta_preview <- fread("data/GSE90496_beta.txt", nrows = 10)
# head(beta_preview)
# 
# colnames(beta_preview)
# 
# num_cols <- ncol(fread("data/GSE90496_beta.txt", nrows = 1))
# print(num_cols)
# 
# num_rows <- as.integer(system("wc -l < data/GSE90496_beta.txt", intern = TRUE))
# print(num_rows)

#dim 5603x 428800 (2801 samples x 428799 probes)
# The following filtering criteria were applied: removal of probes targeting the
# X and Y chromosomes (n= 11,551), removal of probes containing a single-nucleotide
# polymorphism (dbSNP132 Common) within five base pairs of and
# including the targeted CpG site (n
# = 7,998), probes not mapping uniquely to the
# human reference genome (hg19) allowing for one mismatch (n
# = 3,965), and
# probes not included on the Illumina EPIC array (n
# = 32,260). In total, 428,799
# probes targeting CpG sites were kept for further analysis.

#先在远程把pvalue列全都去掉
#在mac上用annotation生成一个promotor列表
#用这个列表去远程筛选promotor probes
#变成promoter_beta.txt之后应该可以在mac上读取

library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
anno <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
head(anno)
colnames(anno)

promoter_idx <- grepl("TSS200", anno$UCSC_RefGene_Group)
promoter_ids <- rownames(anno)[promoter_idx]
length(promoter_ids)
#62625 probes in tss200 promoter

writeLines(promoter_ids, "promoter_ids.txt")


#aggregate steps
library(data.table)
beta <- fread("data/promoter_beta_with_header.txt")
dim(beta)

#set rownames
rownames(beta) <- beta$ID_REF
beta$ID_REF <- NULL
dim(beta)

#make sure probes in beta are promoter in anno
anno_promoter <- anno[promoter_idx, ]
common_probes <- intersect(rownames(beta), rownames(anno_promoter))
length(common_probes)

#make sure common_probes and rowname(beta) are identical
all(common_probes %in% rownames(beta))
length(common_probes) == nrow(beta)
identical(common_probes, rownames(beta))

#create a list, for each element(probe) it's a character vector  
gene_info <- anno_promoter[common_probes, "UCSC_RefGene_Name"]
gene_list <- strsplit(gene_info, ";")

#probe → gene 
#remove "" and duplicates for each element in gene_list
gene_list <- lapply(gene_list, function(x) {
  unique(x[x != ""])
})

# unlist gene_list and create DT (probe x gene) 
probe_gene <- data.table(
  probe = rep(common_probes, sapply(gene_list, length)),
  gene  = unlist(gene_list)
)[gene != ""]

# check the probes that map to multiple genes
multi_gene_probe <- probe_gene[, .N, by = probe][N > 1]
nrow(multi_gene_probe)

# convert df beta to matrix
beta_mat0 <- as.matrix(beta)

# give back rownames and align matrix with probe_gene nrow = 55259 --> 67714
rownames(beta_mat0) <- rownames(beta)
beta_mat <- beta_mat0[probe_gene$probe, ]

# calculate mean gene methylation B value
gene_factor <- factor(
  probe_gene$gene,
  levels = unique(probe_gene$gene)
)

gene_matrix <- rowsum(beta_mat, group = gene_factor) /
  as.vector(table(gene_factor))

# write gene_matrix into a txt file
gene_dt <- as.data.table(gene_matrix, keep.rownames = "Gene")
fwrite(gene_dt, "data/gene_promoter_matrix.txt", sep = "\t")
