#!/usr/bin/env Rscript
# Supplementary Figure 6: Remaining 3 ProteinGym DMS Crystal 3D Sensitivity Maps (1ND4, 1DCT, 1C0P)
# + Remaining 2 Challenging Enzymes from Fig. 8 (Q6UDJ5 & Q8V3U0)
# Strict Arial font, unified calm palette, native 3D aspect ratio preserved, readable 7 pt annotations, no subpanel titles (a-e only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")
suppressPackageStartupMessages({
  library(png)
  library(grid)
})

SUPP_STRUCT <- "/Users/chenchen/Paper/ZymeBind/structures/supp"
OUT_DIR     <- "/Users/chenchen/Paper/ZymeBind/figures/supplementary"

make_3d_card <- function(png_file, top_txt, bot_txt) {
  img <- png::readPNG(file.path(SUPP_STRUCT, png_file))
  g   <- grid::rasterGrob(img, interpolate = TRUE)
  ggplot() +
    annotation_custom(g, xmin = 0.01, xmax = 0.99, ymin = 0.14, ymax = 0.86) +
    annotate("text", x = 0.50, y = 0.930, label = top_txt, size = 2.05, family = "Arial", fontface = "bold", color = "#1E293B", hjust = 0.5, lineheight = 0.95) +
    annotate("text", x = 0.50, y = 0.062, label = bot_txt, size = 1.92, family = "Arial", color = "#334155", hjust = 0.5, lineheight = 0.95) +
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
  "sf6_a_APH3_1ND4_tight.png",
  "APH(3')-II (PDB 1ND4, EC 2.7.1.95)\nPer-residue rho = 0.757 | Active 2.55x",
  "Blue-to-terracotta: sensitivity\nCyan spheres: active-site residues"
)
p_b <- make_3d_card(
  "sf6_b_MHaeIII_1DCT_tight.png",
  "M.HaeIII (PDB 1DCT, EC 2.1.1.37)\nPer-residue rho = 0.579 | Active 2.71x",
  "Blue-to-terracotta: sensitivity\nCyan spheres: catalytic loop"
)
p_c <- make_3d_card(
  "sf6_c_DAOx_1C0P_tight.png",
  "DAOx (PDB 1C0P, EC 1.4.3.3)\nPer-residue rho = 0.501 | AUROC = 0.660",
  "Blue-to-terracotta: sensitivity\nTranslucent surface: PDB 1C0P"
)
p_d <- make_3d_card(
  "sf6_d_Q6UDJ5_tight.png",
  "PsHV-1 Protease Q6UDJ5 (EC 3.4.19.12, 24.7% identity)\nPredicted P = 0.980 | Outperforms 3/5 baselines",
  "Steel-blue: AlphaFold structure\nInset: Catalytic triad Cys30 / His173 / Asp175 (cyan)"
)
p_e <- make_3d_card(
  "sf6_e_Q8V3U0_tight.png",
  "ISAV Esterase Q8V3U0 (EC 3.1.1.53, 0% identity)\nPredicted P = 0.761 | Zero MMseqs2 hit",
  "Slate-teal: AlphaFold structure\nInset: Catalytic triad Ser32 / Asp261 / His264 (cyan)"
)

row1 <- (p_a | p_b | p_c) + plot_layout(widths = c(1, 1, 1))
row2 <- (p_d | p_e) + plot_layout(widths = c(1, 1))
sfig6 <- (row1 / row2) +
  plot_layout(heights = c(0.95, 1.15)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 170 / 25.4
out_p <- file.path(OUT_DIR, "SuppFig6_dms_and_hard_cases_3d_gallery")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(sfig6); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(sfig6); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(sfig6); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(sfig6); grDevices::dev.off()
cat("[EXPORTED] SuppFig6 (183 x 170 mm, Arial)\n")
