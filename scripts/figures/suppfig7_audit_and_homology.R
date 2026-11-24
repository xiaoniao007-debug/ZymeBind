#!/usr/bin/env Rscript
# Supplementary Figure 7: Leak-Free 40%-Cluster Data Hygiene Audit & Metagenome Predicted-Enzyme Homology Spectrum
# Strict Arial font, unified calm palette, no subpanel titles (a-b only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
OUT_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures/supplementary"

aud <- fromJSON(file.path(DATA_DIR, "common_clean_eval_audit.json"), simplifyVector = TRUE)
cs1 <- fromJSON(file.path(DATA_DIR, "case_study_1_metagenome_5000_discovery.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"

# ---- Panel a: Strict Leak-Free Filtering Audit across Evaluation Sources ----
srcs <- c("m2_val", "m2_test", "m1_val", "m1_test")
src_lbls <- c("Main pool\n(validation)", "Main pool\n(blind test)", "Holdout pool\n(validation)", "Holdout pool\n(blind test)")
df_a <- do.call(rbind, lapply(seq_along(srcs), function(i) {
  s <- aud$per_source[[srcs[i]]]
  tot <- s$n
  data.frame(
    src = src_lbls[i],
    cat = c("Retained (leak-free)", "Dropped: 40%-cluster overlap", "Dropped: Accession/seq overlap", "Dropped: Incomplete EC"),
    pct = 100 * c(s$kept, s$drop_cluster_has_train, s$drop_acc_in_train + s$drop_seq_in_train, s$drop_no_ec4) / tot,
    n_kept = s$kept,
    stringsAsFactors = FALSE
  )
}))
df_a$src <- factor(df_a$src, levels = src_lbls)
df_a$cat <- factor(df_a$cat, levels = c("Dropped: Incomplete EC", "Dropped: Accession/seq overlap", "Dropped: 40%-cluster overlap", "Retained (leak-free)"))

p_a <- ggplot(df_a, aes(x = src, y = pct, fill = cat)) +
  geom_col(width = 0.62, color = "#FFFFFF", linewidth = 0.45) +
  geom_text(data = df_a %>% filter(cat == "Retained (leak-free)"),
            aes(x = src, y = pmax(pct / 2, 4.5), label = sprintf("n=%d", n_kept)),
            size = 1.85, family = "Arial", fontface = "bold", color = "#FFFFFF") +
  scale_fill_manual(values = c(
    "Retained (leak-free)"           = COL_HERO,
    "Dropped: 40%-cluster overlap"   = "#CBD5E1",
    "Dropped: Accession/seq overlap" = "#94A3B8",
    "Dropped: Incomplete EC"         = "#E2E8F0"
  )) +
  scale_y_continuous(limits = c(0, 104), expand = c(0, 0), breaks = seq(0, 100, 25)) +
  labs(x = "Candidate evaluation pool", y = "Proportion of candidate proteins (%)", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5.0, family = "Arial")) +
  guides(fill = guide_legend(nrow = 2))

# ---- Panel b: Homology Distribution of All 414 Predicted Metagenome Enzymes ----
hd <- cs1$homology_distribution_of_predicted_enzymes
df_b <- data.frame(
  bin = factor(c("0% identity\n(no MMseqs2 hit)", "1-30% identity\n(twilight zone)", "30-40% identity\n(remote homolog)", ">=40% identity\n(homolog)"),
               levels = c("0% identity\n(no MMseqs2 hit)", "1-30% identity\n(twilight zone)", "30-40% identity\n(remote homolog)", ">=40% identity\n(homolog)")),
  count = c(hd$pid_0_nohit, hd$pid_0_30, hd$pid_30_40, hd$pid_ge40),
  col = c(COL_TERRA, COL_TEAL, COL_SKY, COL_SLATE),
  stringsAsFactors = FALSE
)
df_b$pct <- 100 * df_b$count / sum(df_b$count)

p_b <- ggplot(df_b, aes(x = bin, y = count, fill = col)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%d (%.1f%%)", count, pct)), vjust = -0.4,
            size = 1.95, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_fill_identity() +
  scale_y_continuous(limits = c(0, 248), expand = c(0, 0), breaks = seq(0, 240, 60)) +
  labs(x = "Maximum sequence identity to training set", y = "Predicted metagenome enzymes (n = 414)") +
  theme(axis.text.x = element_text(size = 5.3, family = "Arial"))

sfig7 <- (p_a | p_b) +
  plot_layout(widths = c(1.05, 1.0)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(sfig7, file.path(OUT_DIR, "SuppFig7_data_audit_and_metagenome_homology"), width_mm = 183, height_mm = 86, panel_ids = c("a", "b"))
