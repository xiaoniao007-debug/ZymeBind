#!/usr/bin/env Rscript
# Figure 10: Mutational Fitness Landscape, Metagenome-Scale Throughput & Zero-Shot EC4 Rescues
# Strict Arial font, enlarged 7 pt 3D labels in Panel b, zero Mode 1 / E2E / Ref jargon, no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")
library(png)
library(grid)

DATA_DIR   <- "/Users/chenchen/Paper/ZymeBind/data"
STRUCT_DIR <- "/Users/chenchen/Paper/ZymeBind/structures"
FIG_DIR    <- "/Users/chenchen/Paper/ZymeBind/figures"

dms  <- fromJSON(file.path(DATA_DIR, "ext2c_token_vs_pooled_dms.json"), simplifyVector = TRUE)
ext4 <- fromJSON(file.path(DATA_DIR, "ext4_scalability_and_mode1_cases.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"

# ---- Panel a: Mirrored Per-Residue TEM-1 Catalytic Sensitivity Profile ----
blat <- dms$benchmarks[dms$benchmarks$id == "BLAT_ECOLX_Stiffler_2015", ]
pr   <- blat$per_residue[[1]]
df_a <- data.frame(pos = pr$pos + 2, dms_lof = -pr$mean_dms_score, plm_sens = -pr$mean_mm_score, is_act = pr$is_active)
df_a$dms_norm <- pmax(0, df_a$dms_lof) / max(df_a$dms_lof)
df_a$plm_norm <- pmax(0, df_a$plm_sens) / max(df_a$plm_sens)

cat_motifs <- data.frame(pos = c(70, 130, 166, 234), lbl = c("STFK\n(Ser70)", "SDN\n(Ser130)", "EPELN\n(Glu166)", "KTG\n(Lys234)"))

p_a <- ggplot(df_a, aes(x = pos)) +
  geom_rect(data = cat_motifs, aes(xmin = pos - 3, xmax = pos + 3, ymin = -1.08, ymax = 1.15),
            inherit.aes = FALSE, fill = "#00C4CC", alpha = 0.16) +
  geom_area(aes(y =  dms_norm), fill = COL_HERO, alpha = 0.62) +
  geom_line(aes(y =  dms_norm), color = "#1E3A5F", linewidth = 0.35) +
  geom_area(aes(y = -plm_norm), fill = COL_TERRA, alpha = 0.62) +
  geom_line(aes(y = -plm_norm), color = "#9A482A", linewidth = 0.35) +
  geom_hline(yintercept = 0, color = "#1E293B", linewidth = 0.45) +
  geom_text(data = cat_motifs, aes(x = pos, y = 1.07, label = lbl),
            inherit.aes = FALSE, size = 1.72, family = "Arial", fontface = "bold", color = "#0E7490", lineheight = 0.88) +
  annotate("text", x = 26, y =  0.86, label = "Experimental DMS\nloss-of-function", hjust = 0, size = 1.85, family = "Arial", fontface = "bold", color = "#1E3A5F", lineheight = 0.9) +
  annotate("text", x = 26, y = -0.82, label = "ZymeBind residue\nmutational sensitivity", hjust = 0, size = 1.85, family = "Arial", fontface = "bold", color = "#9A482A", lineheight = 0.9) +
  scale_x_continuous(limits = c(22, 292), expand = c(0, 0), breaks = seq(50, 250, 50)) +
  scale_y_continuous(limits = c(-1.12, 1.24), breaks = c(-1, -0.5, 0, 0.5, 1), labels = c("1.0", "0.5", "0", "0.5", "1.0")) +
  labs(x = "Residue position (amino acids)", y = "Normalized sensitivity")

# ---- Panel b: TEM-1 (1BTL) Pocket-Facing 3D Sensitivity + Enlarged Readable 7 pt Labels ----
img10 <- readPNG(file.path(STRUCT_DIR, "plate_10_tem1_dms_sensitivity_tight.png"))
g10   <- rasterGrob(img10, interpolate = TRUE)
p_b <- ggplot() +
  annotation_custom(g10, xmin = 0.02, xmax = 0.98, ymin = 0.14, ymax = 0.85) +
  annotate("text", x = 0.50, y = 0.930,
           label = "TEM-1 beta-lactamase (PDB 1BTL, EC 3.5.2.6)\nPer-residue Spearman rho = 0.769 | AUROC = 0.883",
           size = 2.25, family = "Arial", fontface = "bold", color = "#1E293B", lineheight = 0.96) +
  annotate("text", x = 0.50, y = 0.060,
           label = "Blue-to-terracotta: mutational sensitivity\nCyan spheres: Ser70 / Lys73 / Ser130 / Glu166 / Lys234",
           size = 2.08, family = "Arial", color = "#334155", lineheight = 0.96) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void(base_family = "Arial") +
  theme(
    panel.border = element_rect(colour = "#CBD5E1", fill = NA, linewidth = 0.45),
    plot.margin = margin(4, 4, 4, 4, "pt"),
    plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
  )

# ---- Panels c & d: 4-Enzyme ProteinGym DMS Matrix + Aligned Active-Site Fold Enrichment ----
bm <- dms$benchmarks
short_nm <- c("TEM-1 beta-lactamase" = "TEM-1", "DNA methyltransferase M.HaeIII" = "M.HaeIII",
              "Aminoglycoside kinase APH(3')-II" = "APH(3')-II", "D-amino acid oxidase (DAOx)" = "DAOx")
enz_lbls <- sprintf("%s (EC %s, n = %s)", unname(short_nm[bm$name]), bm$ec4, format(bm$n_mutants_total, big.mark = ","))
# Order by Mutant Spearman descending: TEM-1, APH(3')-II, M.HaeIII, DAOx
ord_idx <- order(-bm$token_mm_spearman_all)
bm <- bm[ord_idx, ]
enz_lbls <- enz_lbls[ord_idx]
df_c <- data.frame(
  enz = factor(enz_lbls, levels = rev(enz_lbls)),
  `Mutant\nSpearman`      = bm$token_mm_spearman_all,
  `Per-residue\nSpearman` = bm$token_mm_spearman_per_res,
  `LoF\nAUROC`            = bm$token_mm_auroc_lof,
  `Wild-type\nP(EC4)`     = bm$pooled_ec_wt_prob,
  check.names = FALSE
)
df_c_long <- pivot_longer(df_c, cols = -enz, names_to = "metric", values_to = "val")
df_c_long$metric <- factor(df_c_long$metric, levels = c("Mutant\nSpearman", "Per-residue\nSpearman", "LoF\nAUROC", "Wild-type\nP(EC4)"))

p_c1 <- ggplot(df_c_long, aes(x = metric, y = enz, fill = val)) +
  geom_tile(color = "#FFFFFF", linewidth = 0.9) +
  geom_text(aes(label = sprintf("%.3f", val), color = val > 0.68), size = 2.05, family = "Arial", fontface = "bold", show.legend = FALSE) +
  scale_fill_gradientn(colors = c("#EEF5FB", "#B7D1EA", "#5B8DBE", "#2B5C8F"), limits = c(0.35, 1.00)) +
  scale_color_manual(values = c("FALSE" = "#1E293B", "TRUE" = "#FFFFFF")) +
  labs(x = NULL, y = NULL, fill = "Score") +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "right", legend.key.width = unit(5.5, "pt"), legend.key.height = unit(22, "pt"),
        axis.text.x = element_text(size = 5.6, family = "Arial"), axis.text.y = element_text(size = 5.6, family = "Arial"))

df_c2 <- data.frame(
  enz  = factor(enz_lbls, levels = rev(enz_lbls)),
  fold = bm$token_mm_active_fold,
  pval = bm$token_mm_active_mwu_p
)
df_c2$lbl <- ifelse(df_c2$pval < 0.01, sprintf("%.2fx (p = %.1e)", df_c2$fold, df_c2$pval), sprintf("%.2fx (n.s.)", df_c2$fold))

p_c2 <- ggplot(df_c2, aes(x = fold, y = enz, fill = pval < 0.01)) +
  geom_vline(xintercept = 1.0, linetype = "dashed", color = "#94A3B8", linewidth = 0.4) +
  geom_col(width = 0.56) +
  geom_text(aes(label = lbl), hjust = -0.06, size = 1.85, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = c("TRUE" = COL_TERRA, "FALSE" = "#CBD5E1"), guide = "none") +
  scale_x_continuous(limits = c(0, 4.70), expand = c(0, 0), breaks = c(0, 1, 2, 3)) +
  labs(x = "Active-site fold enrichment", y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.line.y = element_blank())

# ---- Panel e: Log10 Metagenome-Scale Throughput ----
sc <- ext4$scalability
df_d <- data.frame(
  stage = c("ZymeBind Fusion Head (1 GPU)", "ZymeBind 4-Model Head (1 GPU)",
            "MMseqs2 Homology Search (16 CPUs)", "ZymeBind Full Pipeline (4 GPUs)",
            "ESM-2 650M Encoder (1 GPU)", "ZymeBind Full Pipeline (1 GPU)"),
  hours = c(sc$zymebind_head_single$hours_per_1m_1gpu,
            sc$zymebind_head_ens4$hours_per_1m_1gpu,
            sc$mmseqs2_16cpu$hours_per_1m_16cpu,
            sc$zymebind_seq_only_e2e$hours_per_1m_4gpu,
            sc$esm2_650m_fp16$hours_per_1m_1gpu,
            sc$zymebind_seq_only_e2e$hours_per_1m_1gpu),
  grp = c("ZymeBind Head", "ZymeBind Head", "Baseline", "ZymeBind Pipeline", "Baseline", "ZymeBind Pipeline"),
  stringsAsFactors = FALSE
)
df_d$stage <- factor(df_d$stage, levels = rev(df_d$stage))
df_d$lbl <- ifelse(df_d$hours < 0.1, sprintf("%.3f h", df_d$hours), sprintf("%.2f h", df_d$hours))

p_d <- ggplot(df_d, aes(x = hours, y = stage, color = grp)) +
  geom_segment(aes(x = 0.005, xend = hours, y = stage, yend = stage), linewidth = 1.0, alpha = 0.85) +
  geom_point(size = 2.5) +
  geom_text(aes(label = lbl), hjust = -0.18, size = 1.85, family = "Arial", fontface = "bold", show.legend = FALSE) +
  scale_color_manual(values = c("ZymeBind Head" = COL_TEAL, "ZymeBind Pipeline" = COL_HERO, "Baseline" = COL_SLATE)) +
  scale_x_log10(limits = c(0.005, 145), breaks = c(0.01, 0.1, 1, 10, 50), labels = c("0.01", "0.1", "1", "10", "50")) +
  labs(x = "Wall-clock hours per 1M proteins (log10 scale)", y = NULL, color = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5.1, family = "Arial"), axis.text.y = element_text(size = 5.4, family = "Arial"))

# ---- Panel f: Zero-Shot Unseen-EC4 Rank-1 Rescues ----
r1 <- ext4$mode1_zeroshot_cases$top_10_rank1_exact_zeroshot[1:8, ]
df_e <- data.frame(label = sprintf("%s (%s)", r1$accession, r1$true_unseen_ec4),
                   rank_gcn = r1$rank_with_gcn, rank_no = r1$rank_no_gcn, stringsAsFactors = FALSE)
df_e$label <- factor(df_e$label, levels = rev(df_e$label))

p_e <- ggplot(df_e) +
  geom_segment(aes(x = rank_no, xend = rank_gcn, y = label, yend = label), color = "#CBD5E1", linewidth = 1.05) +
  geom_point(aes(x = rank_gcn, y = label, color = "With ontology GCN"), size = 2.4) +
  geom_point(aes(x = rank_no,  y = label, color = "ZymeBind (no GCN)"), size = 2.7) +
  geom_text(aes(x = rank_gcn + 2.5, y = label, label = sprintf("#%d -> #1", rank_gcn)),
            hjust = 0, size = 1.75, family = "Arial", color = COL_HERO, fontface = "bold") +
  annotate("text", x = 66, y = 2.0,
           label = "376 unseen EC4s (906 pairs)\nR@10: 41.3% -> 59.1%\n83 exact Rank-1 hits",
           size = 1.68, family = "Arial", hjust = 0, fontface = "bold", color = "#1E3A5F", lineheight = 1.12) +
  scale_color_manual(values = c("With ontology GCN" = COL_SLATE, "ZymeBind (no GCN)" = COL_TERRA)) +
  scale_x_continuous(limits = c(0, 108), breaks = c(1, 20, 40, 60, 80, 100)) +
  labs(x = "Zero-shot retrieval rank among 5,855 EC4s", y = NULL, color = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5.1, family = "Arial"), axis.text.y = element_text(size = 5.4, family = "Arial"))

row1 <- (free(p_a, type = "label") | p_b) + plot_layout(widths = c(1.15, 1.0))
row2 <- (p_c1 | p_c2) + plot_layout(widths = c(1.45, 0.85))
row3 <- (p_d | p_e) + plot_layout(widths = c(1.0, 1.05))
fig10 <- (row1 / row2 / row3) +
  plot_layout(heights = c(1.16, 0.72, 1.02)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 198 / 25.4
out_p <- file.path(FIG_DIR, "Fig10_dms_scalability_mode1")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(fig10); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(fig10); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(fig10); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(fig10); grDevices::dev.off()
cat("[EXPORTED] Fig10 (183 x 198 mm, Arial, unified nomenclature, enlarged 3D labels)\n")
