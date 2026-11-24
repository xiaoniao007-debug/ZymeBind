#!/usr/bin/env Rscript
# Supplementary Figure 4: 6-Protein 3D Fold Superposition Gallery of 0%-Identity Dark-Proteome Enzymes
# Strict Arial font, unified calm palette, native 3D aspect ratio preserved, readable 7 pt annotations, no subpanel titles (a-f only).

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
  "sf4_a_A0A328P3J7_6S16_tight.png",
  "AF-A0A328P3J7 vs RuvC 6S16 (EC 3.1.21.10)\nNo InterPro | Identity = 12.1% | Prob = 1.000",
  "Steel-blue: AF-A0A328P3J7 (pLDDT 88.2)\nSurface: 6S16 | Inset: catalytic pocket"
)
p_b <- make_3d_card(
  "sf4_b_A0A328P184_4EP4_tight.png",
  "AF-A0A328P184 vs Endonuclease 4EP4 (EC 3.1.21.10)\nNo InterPro | Identity = 11.4% | Prob = 1.000",
  "Steel-blue: AF-A0A328P184 (pLDDT 91.1)\nSurface: 4EP4 | Inset: catalytic pocket"
)
p_c <- make_3d_card(
  "sf4_c_A0A7L6AW24_6WAN_tight.png",
  "AF-A0A7L6AW24 vs Endonuclease 6WAN (EC 3.1.21.4)\nNo InterPro | Identity = 11.7% | Prob = 1.000",
  "Steel-blue: AF-A0A7L6AW24 (pLDDT 83.8)\nSurface: 6WAN | Inset: catalytic pocket"
)
p_d <- make_3d_card(
  "sf4_d_A0AAD7UN86_4RAJ_tight.png",
  "AF-A0AAD7UN86 vs Heme Oxygenase 4RAJ (EC 1.14.14.18)\nEC1 Oxidoreductase | Identity = 23.7% | Prob = 1.000",
  "Slate-teal: AF-A0AAD7UN86 (pLDDT 89.6)\nSurface: 4RAJ | Inset: heme pocket"
)
p_e <- make_3d_card(
  "sf4_e_A0AAD7ULK5_1P9Y_tight.png",
  "AF-A0AAD7ULK5 vs Trigger Factor 1P9Y (EC 5.2.1.8)\nEC5 Isomerase | Identity = 16.8% | Prob = 1.000",
  "Slate-teal: AF-A0AAD7ULK5 (pLDDT 70.9)\nSurface: 1P9Y | Inset: PPIase cleft"
)
p_f <- make_3d_card(
  "sf4_f_A0AAD7UH24_1TSF_tight.png",
  "AF-A0AAD7UH24 vs RNase P 1TSF (EC 3.1.26.5)\nEC3 Hydrolase | Identity = 26.8% | Prob = 1.000",
  "Slate-teal: AF-A0AAD7UH24 (pLDDT 77.2)\nSurface: 1TSF | Inset: catalytic core"
)

sfig4 <- wrap_plots(p_a, p_b, p_c, p_d, p_e, p_f, ncol = 2) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(sfig4, file.path(OUT_DIR, "SuppFig4_dark_proteome_6protein_3d_gallery"), width_mm = 183, height_mm = 228, panel_ids = c("a", "b", "c", "d", "e", "f"))
