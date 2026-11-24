#!/usr/bin/env Rscript
# Supplementary Figure 5: M. jannaschii Archaeal Pathway-Hole 3D Structures (4 proteins) + Whole-Proteome Known-EC4 QC
# Strict Arial font, unified calm palette, native 3D aspect ratio preserved, readable 7 pt annotations, no subpanel titles (a-e only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")
suppressPackageStartupMessages({
  library(png)
  library(grid)
})

DATA_DIR    <- "/Users/chenchen/Paper/ZymeBind/data"
SUPP_STRUCT <- "/Users/chenchen/Paper/ZymeBind/structures/supp"
OUT_DIR     <- "/Users/chenchen/Paper/ZymeBind/figures/supplementary"

ext3 <- fromJSON(file.path(DATA_DIR, "ext3_extremophile_pathway_holes.json"), simplifyVector = TRUE)

make_3d_card <- function(png_file, top_txt, bot_txt) {
  img <- png::readPNG(file.path(SUPP_STRUCT, png_file))
  g   <- grid::rasterGrob(img, interpolate = TRUE)
  ggplot() +
    annotation_custom(g, xmin = 0.02, xmax = 0.98, ymin = 0.14, ymax = 0.86) +
    annotate("text", x = 0.50, y = 0.930, label = top_txt, size = 2.15, family = "Arial", fontface = "bold", color = "#1E293B", hjust = 0.5, lineheight = 0.95) +
    annotate("text", x = 0.50, y = 0.062, label = bot_txt, size = 1.98, family = "Arial", color = "#334155", hjust = 0.5, lineheight = 0.95) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(
      panel.border = element_rect(fill = NA, colour = "#CBD5E1", linewidth = 0.45),
      plot.margin = margin(4, 4, 4, 4, "pt"),
      plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
    )
}

p_a <- make_3d_card(
  "sf5_a_Q59008_tight.png",
  "Q59008 (MJ1613) | Pred EC 3.5.1.90 (P = 0.830)\n0% training identity | Porphyrin / Vitamin B12 (map00860)",
  "Blue gradient: AlphaFold pLDDT\nCyan sticks/spheres: central polar catalytic core"
)
p_b <- make_3d_card(
  "sf5_b_Q58993_tight.png",
  "Q58993 (MJ1598) | Pred EC 2.4.2.21 (P = 0.682)\n0% training identity | Porphyrin / Vitamin B12 (map00860)",
  "Blue gradient: AlphaFold pLDDT\nCyan sticks/spheres: central polar catalytic core"
)
p_c <- make_3d_card(
  "sf5_c_Q57746_tight.png",
  "Q57746 (MJ0298) | Pred EC 1.3.5.3 (P = 0.681)\n0% training identity | Porphyrin / Vitamin B12 (map00860)",
  "Blue gradient: AlphaFold pLDDT\nCyan sticks/spheres: central polar catalytic core"
)
p_d <- make_3d_card(
  "sf5_d_Q58193_tight.png",
  "Q58193 (MJ0783) | Pred EC 3.5.1.9 (P = 0.986)\n27.8% identity | Tryptophan & Glyoxylate (map00380/00630)",
  "Blue gradient: AlphaFold pLDDT\nCyan sticks/spheres: central polar catalytic core"
)

# ---- Panel e: Whole-Proteome Scan Summary & Known-Enzyme Sanity Check (n = 1,786) ----
df_e <- data.frame(
  cat = factor(c("Known EC4\ntop-1 EC4 acc.", "Known EC4\ntop-1 EC3 acc.", "Unannotated\npredicted enzymes", "Pathway-hole\nproteins (8 pathways)"),
               levels = c("Known EC4\ntop-1 EC4 acc.", "Known EC4\ntop-1 EC3 acc.", "Unannotated\npredicted enzymes", "Pathway-hole\nproteins (8 pathways)")),
  val = c(100 * ext3$known_ec4_top1_acc, 100 * ext3$known_ec3_top1_acc,
          100 * ext3$n_unannotated_pred_enzyme_ge_tau062 / ext3$n_unannotated_no_ec,
          100 * ext3$n_pathway_hole_fill_proteins / ext3$n_unannotated_pred_enzyme_ge_tau062),
  lbl = c(sprintf("%.1f%%\n(n = %d)", 100 * ext3$known_ec4_top1_acc, ext3$n_known_complete_ec4),
          sprintf("%.1f%%\n(n = %d)", 100 * ext3$known_ec3_top1_acc, ext3$n_known_complete_ec4),
          sprintf("%.1f%%\n(%d / %d)", 100 * ext3$n_unannotated_pred_enzyme_ge_tau062 / ext3$n_unannotated_no_ec, ext3$n_unannotated_pred_enzyme_ge_tau062, ext3$n_unannotated_no_ec),
          sprintf("%d proteins\n(9 missing EC4s)", ext3$n_pathway_hole_fill_proteins)),
  col = c("#2B5C8F", "#4D908E", "#6B9AC4", "#D97A54"),
  stringsAsFactors = FALSE
)

p_e <- ggplot(df_e, aes(x = cat, y = val, fill = col)) +
  geom_col(width = 0.56) +
  geom_text(aes(label = lbl), vjust = -0.35, size = 1.90, family = "Arial", fontface = "bold", color = "#1E293B", lineheight = 0.90) +
  scale_fill_identity() +
  scale_y_continuous(limits = c(0, 114), expand = c(0, 0), breaks = seq(0, 100, 25)) +
  labs(x = "M. jannaschii DSM 2661 whole-proteome scan (n = 1,786 proteins)", y = "Percentage (%)") +
  theme(axis.text.x = element_text(size = 5.6, family = "Arial"))

sfig5 <- ((p_a | p_b) / (p_c | p_d) / p_e) +
  plot_layout(heights = c(1.08, 1.08, 0.80)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 220 / 25.4
out_p <- file.path(OUT_DIR, "SuppFig5_mja_pathway_holes_3d_and_qc")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(sfig5); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(sfig5); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(sfig5); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(sfig5); grDevices::dev.off()
cat("[EXPORTED] SuppFig5 (183 x 220 mm, Arial)\n")
