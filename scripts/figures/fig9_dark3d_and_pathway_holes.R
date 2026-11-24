#!/usr/bin/env Rscript
# Figure 9: Systematic 3D Fold Validation of Zero-Hit Enzymes & Extremophile KEGG Pathway Reconstruction
# Strict Arial font, unified nomenclature (zero pid tags), zoomed ATP-pocket 3D plate + 7 pt labels, no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")
library(png)
library(grid)

DATA_DIR   <- "/Users/chenchen/Paper/ZymeBind/data"
STRUCT_DIR <- "/Users/chenchen/Paper/ZymeBind/structures"
FIG_DIR    <- "/Users/chenchen/Paper/ZymeBind/figures"

ext1 <- fromJSON(file.path(DATA_DIR, "ext1_dark_metagenome_batch_foldseek_3d.json"), simplifyVector = TRUE)
ext3 <- fromJSON(file.path(DATA_DIR, "ext3_extremophile_pathway_holes.json"), simplifyVector = TRUE)

# ---- Panel a: Systematic Foldseek 3D Validation (0% training identity, n = 23) ----
df_p <- ext1$proteins
df_a <- data.frame(
  acc        = df_p$accession,
  prob3d     = df_p$top1_prob,
  seqId      = df_p$top1_seqId,
  plddt      = df_p$mean_plddt,
  pred_ec    = df_p$pred_ec4,
  pdb_id     = ifelse(is.na(df_p$top1_pdb_id) | df_p$top1_pdb_id == "", "None", df_p$top1_pdb_id),
  ultra_dark = df_p$is_ultra_dark_no_ipr,
  ec_match   = df_p$best_ec_match_top5,
  stringsAsFactors = FALSE
)
df_a$status <- ifelse(
  df_a$prob3d >= 0.90 & df_a$ec_match == "EC4",
  ifelse(df_a$ultra_dark, "De novo (no InterPro): exact EC4 match", "0% identity: exact EC4 match"),
  ifelse(df_a$prob3d >= 0.90, "0% identity: EC2 subclass match", "Low-confidence 3D hit (<0.5)")
)
df_a$status <- factor(df_a$status, levels = c(
  "De novo (no InterPro): exact EC4 match",
  "0% identity: exact EC4 match",
  "0% identity: EC2 subclass match",
  "Low-confidence 3D hit (<0.5)"
))

hl_accs <- c("A0A328P3J7", "A0A232LNA0", "A0AAD7UH24")
df_a$lbl <- ifelse(df_a$acc %in% hl_accs, sprintf("%s (%s)", df_a$acc, df_a$pdb_id), "")
top_idx <- which(df_a$prob3d >= 0.99)
ord_top <- top_idx[order(df_a$seqId[top_idx])]
df_a$prob_plot <- df_a$prob3d
if (length(ord_top) > 1) {
  offsets <- rep(c(-0.045, 0.00, 0.045, -0.022, 0.022), length.out = length(ord_top))
  df_a$prob_plot[ord_top] <- 1.00 + offsets
}

p_a <- ggplot(df_a, aes(x = seqId, y = prob_plot, color = status, shape = ultra_dark)) +
  geom_hline(yintercept = 0.90, linetype = "dashed", color = "#888888", linewidth = 0.35) +
  geom_point(size = 2.4, alpha = 0.92) +
  ggrepel::geom_text_repel(
    aes(label = lbl), size = 1.80, family = "Arial", lineheight = 0.88, box.padding = 0.38, nudge_y = -0.15,
    point.padding = 0.25, max.overlaps = 20, show.legend = FALSE, segment.size = 0.28, seed = 7
  ) +
  annotate("text", x = 10, y = 0.44,
           label = "15/23 (65.2%) Foldseek Prob = 1.000\nEC1/EC2 agreement: 15/15 (100%)\nExact EC4 agreement: 14/15 (93.3%)",
           size = 1.88, family = "Arial", hjust = 0, fontface = "bold", color = "#1E3A5F", lineheight = 1.15) +
  scale_color_manual(values = c(
    "De novo (no InterPro): exact EC4 match" = "#D97A54",
    "0% identity: exact EC4 match"           = "#4D908E",
    "0% identity: EC2 subclass match"        = "#C98B63",
    "Low-confidence 3D hit (<0.5)"           = "#8D99AE"
  )) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), guide = "none") +
  scale_x_continuous(trans = "sqrt", limits = c(0, 66), breaks = c(0, 5, 10, 15, 20, 30, 45, 60)) +
  scale_y_continuous(limits = c(-0.04, 1.08), breaks = seq(0, 1.0, 0.25)) +
  labs(x = "RCSB PDB structural sequence identity (%, square-root scale)", y = "Foldseek 3D alignment probability", color = NULL) +
  theme(legend.position = "top", legend.direction = "vertical",
        legend.text = element_text(size = 5.2, family = "Arial"), legend.key.height = unit(6.5, "pt"))

# ---- Panel b: Pocket-Facing Ultra-Dark 3D Kinase Superposition + Zoomed ATP Cleft & 7 pt Labels ----
img9 <- readPNG(file.path(STRUCT_DIR, "plate_9_ultradark_kinase_6qav_tight.png"))
g9 <- rasterGrob(img9, interpolate = TRUE)
p_b <- ggplot() +
  annotation_custom(g9, xmin = 0.01, xmax = 0.99, ymin = 0.13, ymax = 0.87) +
  annotate("text", x = 0.50, y = 0.935,
           label = "AF-A0A232LNA0 vs ULK2 Kinase 6QAV (EC 2.7.11.1)\n0% training identity, no InterPro | Foldseek Prob = 1.000",
           size = 2.25, family = "Arial", fontface = "bold", color = "#1E293B", lineheight = 0.96) +
  annotate("text", x = 0.50, y = 0.060,
           label = "Steel-blue: AF-A0A232LNA0 | Surface: 6QAV crystal\nInset: ATP-pocket ligand (orange) + Glu33/Lys142/Asn214 (cyan)",
           size = 2.10, family = "Arial", color = "#334155", lineheight = 0.96) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void(base_family = "Arial") +
  theme(
    panel.border = element_rect(colour = "#CBD5E1", fill = NA, linewidth = 0.45),
    plot.margin = margin(34, 4, 4, 4, "pt"),
    plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
  )

# ---- Panel c: Topological KEGG Pathway Reconstruction (M. jannaschii) ----
topo_nodes <- data.frame(
  x = c(1.1, 3.3, 5.6, 7.8, 9.2,   2.2, 5.6, 8.6),
  y = c(7.8, 7.8, 7.8, 7.8, 7.8,   5.7, 5.7, 5.7),
  type = c("met", "hole", "met", "hole", "met", "met", "hole", "met"),
  lbl = c("Adenosyl-\ncobinamide", "3.5.1.90\nQ59008", "Cobyrinate", "2.4.2.21\nQ58993", "Vitamin\nB12",
          "Protoporph.\nIX-gen", "1.3.5.3\nQ57746", "Protoporph.\nIX"),
  stringsAsFactors = FALSE
)
topo_edges <- data.frame(
  x = c(1.65, 3.85, 6.15, 8.35,  2.75, 6.15),
  xend = c(2.70, 5.00, 7.20, 8.75,  5.00, 8.05),
  y = c(7.8, 7.8, 7.8, 7.8,  5.7, 5.7),
  yend = c(7.8, 7.8, 7.8, 7.8,  5.7, 5.7)
)

pws_df <- data.frame(
  pw = c("Porphyrin / Vitamin B12", "Pyruvate metabolism", "Tryptophan metabolism", "Glyoxylate metabolism",
         "Amino/nucleotide sugars", "Val / Leu / Ile", "Oxidative phosph.", "Selenocompound"),
  n_new = c(3, 1, 1, 1, 1, 1, 1, 1),
  dark  = c(TRUE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE),
  stringsAsFactors = FALSE
)
pws_df$y <- seq(3.9, 0.7, length.out = nrow(pws_df))
pws_df$xend <- 3.6 + pws_df$n_new * 1.75

p_c <- ggplot() +
  annotate("rect", xmin = 0.15, xmax = 9.85, ymin = 4.75, ymax = 9.55, fill = "#FAF4F1", color = "#E2E8F0", linewidth = 0.4) +
  annotate("text", x = 0.4, y = 9.22, label = "KEGG map00860 (Porphyrin / Vitamin B12): +3 zero-hit enzymatic steps",
           hjust = 0, size = 1.75, family = "Arial", fontface = "bold", color = "#9A482A") +
  geom_segment(data = topo_edges, aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(3.5, "pt"), type = "closed"), color = "#64748B", linewidth = 0.55) +
  geom_point(data = topo_nodes[topo_nodes$type == "met", ], aes(x = x, y = y),
             shape = 21, size = 6.2, fill = "#FFFFFF", color = "#2B5C8F", stroke = 0.7) +
  geom_point(data = topo_nodes[topo_nodes$type == "hole", ], aes(x = x, y = y),
             shape = 21, size = 7.8, fill = "#D97A54", color = "#FFFFFF", stroke = 0.7) +
  geom_text(data = topo_nodes[topo_nodes$type == "met", ], aes(x = x, y = y - 0.85, label = lbl),
            size = 1.50, family = "Arial", color = "#334155", lineheight = 0.86) +
  geom_text(data = topo_nodes[topo_nodes$type == "hole", ], aes(x = x, y = y, label = lbl),
            size = 1.45, family = "Arial", color = "#FFFFFF", fontface = "bold", lineheight = 0.86) +
  annotate("text", x = 0.25, y = 4.42, label = "Reconstructed M. jannaschii pathways (orange: <30% sequence identity)",
           hjust = 0, size = 1.72, family = "Arial", fontface = "bold", color = "#1E293B") +
  geom_text(data = pws_df, aes(x = 3.45, y = y, label = pw), hjust = 1, size = 1.60, family = "Arial", color = "#334155") +
  geom_segment(data = pws_df, aes(x = 3.6, xend = xend, y = y, yend = y,
                                  color = ifelse(dark, "#D97A54", "#2B5C8F")), linewidth = 2.6, lineend = "round") +
  geom_text(data = pws_df, aes(x = xend + 0.25, y = y, label = paste0("+", n_new)),
            hjust = 0, size = 1.62, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_color_identity() +
  scale_x_continuous(limits = c(0, 10), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.2, 9.7), expand = c(0, 0)) +
  theme_void(base_family = "Arial") +
  theme(
    plot.margin = margin(4, 4, 4, 4, "pt"),
    plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
  )

# ---- Panel d: Newly Assigned Archaeal Pathway-Hole Enzymes (n = 12) ----
fills <- ext3$top_25_pathway_hole_fills
df_d <- data.frame(
  acc = fills$accession,
  ec = fills$pred_ec4,
  prob = fills$pred_prob,
  pid = fills$max_train_pident,
  stringsAsFactors = FALSE
)
df_d$regime <- ifelse(df_d$pid == 0, "0% identity (no MMseqs2 hit)",
               ifelse(df_d$pid < 30, "Twilight zone (<30% identity)", "Moderate identity (30-50%)"))
df_d$regime <- factor(df_d$regime, levels = c("0% identity (no MMseqs2 hit)", "Twilight zone (<30% identity)", "Moderate identity (30-50%)"))
df_d$lbl <- sprintf("%s\n(%s)", df_d$acc, df_d$ec)

p_d <- ggplot(df_d, aes(x = pid, y = prob, color = regime)) +
  geom_vline(xintercept = 30, linetype = "dashed", color = "#888888", linewidth = 0.35) +
  geom_point(size = 2.6, alpha = 0.92) +
  ggrepel::geom_text_repel(aes(label = lbl), size = 1.72, family = "Arial", lineheight = 0.88, max.overlaps = 15,
                           box.padding = 0.22, show.legend = FALSE, segment.size = 0.25, seed = 42) +
  scale_color_manual(values = c("0% identity (no MMseqs2 hit)"  = "#D97A54",
                                "Twilight zone (<30% identity)" = "#4D908E",
                                "Moderate identity (30-50%)"    = "#2B5C8F")) +
  scale_x_continuous(limits = c(-3, 54), breaks = seq(0, 50, 10)) +
  scale_y_continuous(limits = c(0.62, 1.04), breaks = seq(0.65, 1.00, 0.1)) +
  labs(x = "Maximum sequence identity to training set (%)", y = "ZymeBind (Sequence only) probability", color = NULL) +
  theme(legend.position = "top", legend.direction = "vertical",
        legend.text = element_text(size = 5.2, family = "Arial"), legend.key.height = unit(6.5, "pt"))

fig9 <- wrap_plots(p_a, p_b, p_c, p_d, ncol = 2, heights = c(1.04, 1.0)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(
  fig9,
  file.path(FIG_DIR, "Fig9_dark3d_and_pathway_holes"),
  width_mm = 183, height_mm = 166,
  panel_ids = c("a", "b", "c", "d")
)
