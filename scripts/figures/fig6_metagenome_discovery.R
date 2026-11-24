#!/usr/bin/env Rscript
# Figure 6: Metagenomic Dark-Proteome Enzyme Discovery (n = 4,281)
# Strict Arial font, free(p_b) centering, non-overlapping lollipop labels, enlarged 7 pt 3D labels, no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")
suppressPackageStartupMessages({
  library(png)
  library(grid)
})

DATA_DIR   <- "/Users/chenchen/Paper/ZymeBind/data"
STRUCT_DIR <- "/Users/chenchen/Paper/ZymeBind/structures"
FIG_DIR    <- "/Users/chenchen/Paper/ZymeBind/figures"

cs1 <- fromJSON(file.path(DATA_DIR, "case_study_1_metagenome_5000_discovery.json"), simplifyVector = TRUE)

make_3d_panel <- function(png_path, top_txt, bot_txt) {
  img <- png::readPNG(png_path)
  g <- grid::rasterGrob(img, interpolate = TRUE)
  ggplot() +
    annotation_custom(g, xmin = 0.01, xmax = 0.99, ymin = 0.13, ymax = 0.87) +
    annotate("text", x = 0.5, y = 0.935, label = top_txt, size = 2.30, family = "Arial", fontface = "bold", color = "#1E293B", hjust = 0.5, lineheight = 0.96) +
    annotate("text", x = 0.5, y = 0.060, label = bot_txt, size = 2.15, family = "Arial", color = "#334155", hjust = 0.5, lineheight = 0.96) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    labs(x = "", y = NULL) +
    theme_nature() +
    theme(
      axis.line = element_blank(), axis.ticks = element_blank(),
      axis.text = element_blank(),
      panel.border = element_rect(fill = NA, colour = "#CBD5E1", linewidth = 0.45)
    )
}

# ---- Panel a: Smooth Cubic Bezier Alluvial Discovery Flow ----
bezier_ribbon <- function(x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, id, fill_col, n = 40) {
  t <- seq(0, 1, length.out = n)
  s <- 3 * t^2 - 2 * t^3
  xt <- x0 + (x1 - x0) * t
  yt_hi <- y0_hi + (y1_hi - y0_hi) * s
  yt_lo <- y0_lo + (y1_lo - y0_lo) * s
  data.frame(x = c(xt, rev(xt)), y = c(yt_hi, rev(yt_lo)), id = id, fill_col = fill_col, stringsAsFactors = FALSE)
}

nodes_s <- data.frame(
  x = c(1, 2, 2, 3, 3, 4, 4),
  ymin = c(0, 0, 82, 0, 48, 23, 0),
  ymax = c(100, 78, 100, 42, 78, 43, 20),
  col = c("#64748B", "#475569", "#CBD5E1", "#4D908E", "#E2E8F0", "#2B5C8F", "#D97A54"),
  tcol = c("#FFFFFF", "#FFFFFF", "#1E293B", "#FFFFFF", "#1E293B", "#FFFFFF", "#FFFFFF"),
  lbl = c("Unannot.\n4,281", "No hit\n3,503", "Homolog\n778",
          "ZymeBind\n213", "Other\n3,290",
          "AFDB 3D\n138", "De novo\n131"),
  stringsAsFactors = FALSE
)

ribbons <- rbind(
  bezier_ribbon(1.26, 1.74, 0, 78, 0, 78, "r1", "#94A3B8"),
  bezier_ribbon(1.26, 1.74, 78, 100, 82, 100, "r2", "#CBD5E1"),
  bezier_ribbon(2.26, 2.74, 0, 42, 0, 42, "r3", "#4D908E"),
  bezier_ribbon(2.26, 2.74, 42, 78, 48, 78, "r4", "#E2E8F0"),
  bezier_ribbon(3.26, 3.74, 21, 42, 23, 43, "r5", "#2B5C8F"),
  bezier_ribbon(3.26, 3.74, 0, 21, 0, 20, "r6", "#D97A54")
)

p_a <- ggplot() +
  geom_polygon(data = ribbons, aes(x = x, y = y, group = id, fill = fill_col), alpha = 0.36) +
  geom_rect(data = nodes_s, aes(xmin = x - 0.26, xmax = x + 0.26, ymin = ymin, ymax = ymax, fill = col),
            color = "#FFFFFF", linewidth = 0.5) +
  geom_text(data = nodes_s, aes(x = x, y = 0.5 * (ymin + ymax), label = lbl, color = tcol),
            size = 1.55, family = "Arial", fontface = "bold", lineheight = 0.88) +
  scale_fill_identity() +
  scale_color_identity() +
  scale_x_continuous(limits = c(0.68, 4.32), breaks = 1:4,
                     labels = c("Proteome", "Homology", "ZymeBind", "Validation")) +
  scale_y_continuous(limits = c(-4, 105), expand = c(0, 0)) +
  labs(x = " ", y = NULL) +
  theme_nature() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), axis.text.y = element_blank(),
        axis.line.x = element_line(color = NA), axis.ticks.x = element_line(color = NA),
        axis.text.x = element_text(size = 5.8, family = "Arial", face = "bold", color = "#334155"))

# ---- Panel b: Nightingale Rose Chart of All 7 EC1 Classes (properly centered in column 2!) ----
ec1_labels <- c("EC1" = "EC1\nOxidored.", "EC2" = "EC2\nTransferases", "EC3" = "EC3\nHydrolases",
                "EC4" = "EC4\nLyases", "EC5" = "EC5\nIsomerases", "EC6" = "EC6\nLigases", "EC7" = "EC7\nTranslocases")
df_b <- data.frame(
  ec1 = names(ec1_labels),
  label = unname(ec1_labels),
  count = sapply(names(ec1_labels), function(k) as.numeric(cs1$ec1_distribution_dark_lt30[[k]])),
  stringsAsFactors = FALSE
)
df_b <- df_b[order(-df_b$count), ]
df_b$label <- factor(df_b$label, levels = df_b$label)
df_b$r_val <- sqrt(df_b$count)

rose_pal <- c("#2B5C8F", "#4D908E", "#6B9AC4", "#D97A54", "#8D99AE", "#A4C2E0", "#E3B08D")

p_b <- ggplot(df_b, aes(x = label, y = r_val, fill = label)) +
  geom_col(width = 0.90, color = "#FFFFFF", linewidth = 0.5, alpha = 0.92) +
  geom_text(aes(y = r_val + 1.45, label = count), size = 2.05, family = "Arial", fontface = "bold", color = "#1E293B") +
  coord_polar(start = -pi / 7, clip = "off") +
  scale_fill_manual(values = rose_pal, guide = "none") +
  scale_y_continuous(limits = c(0, 13.0), breaks = sqrt(c(10, 50, 100))) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_family = "Arial") +
  theme(
    axis.text.y = element_blank(), axis.ticks = element_blank(),
    panel.grid.major = element_line(color = "#E2E8F0", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 5.4, family = "Arial", face = "bold", color = "#1E293B", lineheight = 0.88),
    plot.margin = margin(2, 18, 2, -14, "pt"),
    plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
  )

# ---- Panel c: Lollipop Chart of Top 10 Dark-Proteome EC3 Sub-subclasses (shifted count labels, zero overlap!) ----
top_ec3 <- cs1$top_15_ec3_dark_lt30
df_c <- data.frame(
  ec3 = paste0("EC ", top_ec3[1:10, 1]),
  count = as.numeric(top_ec3[1:10, 2]),
  ec1 = paste0("EC", sub("\\..*", "", top_ec3[1:10, 1])),
  stringsAsFactors = FALSE
)
df_c$ec3 <- factor(df_c$ec3, levels = rev(df_c$ec3))

p_c <- ggplot(df_c, aes(x = count, y = ec3, color = ec1)) +
  geom_segment(aes(x = 0, xend = count, y = ec3, yend = ec3), linewidth = 0.85, alpha = 0.82) +
  geom_point(aes(size = count), alpha = 0.95) +
  geom_text(aes(x = count + 2.4, label = count), hjust = 0, size = 2.0, family = "Arial", fontface = "bold", show.legend = FALSE) +
  scale_color_manual(values = c("EC2" = "#2B5C8F", "EC3" = "#4D908E", "EC7" = "#D97A54"), guide = "none") +
  scale_size_continuous(range = c(2.0, 3.8), guide = "none") +
  scale_x_continuous(limits = c(0, 54), expand = c(0, 0), breaks = seq(0, 50, 10)) +
  labs(x = "Discovered enzymes (sequence identity < 30%)", y = NULL)

# ---- Panel d: Blind EBI InterPro Corroboration (0% sequence identity) ----
df_d <- data.frame(
  label = c(
    "A0AAD7ULK5 | EC 5.2.1.8\nIPR036611 Trigger factor",
    "A0A232LXT3 | EC 2.3.2.27\nIPR038886 E3 ubiquitin ligase",
    "A0A232LPP5 | EC 3.1.26.4\nIPR052055 RNase H domain",
    "A0AAD7UH24 | EC 3.1.26.5\nIPR016848 RNase P subunit",
    "A0A232LRB2 | EC 7.1.1.2\nIPR008699 NADH oxidored.",
    "A0AAD7ULU7 | EC 2.5.1.18\nIPR001129 MAPEG transferase",
    "A0AAD7U9D2 | EC 3.2.1.14\nIPR017853 Glycoside hydrolase",
    "A0ACC3SFP4 | EC 3.1.26.5\nNo InterPro / Pfam domain"
  ),
  prob = c(1.000, 0.9994, 0.9999, 0.9987, 0.9986, 0.9934, 0.9803, 1.000),
  grp  = c(rep("Consistent InterPro signature", 7), "De novo (no domain signature)"),
  stringsAsFactors = FALSE
)
df_d$label <- factor(df_d$label, levels = rev(df_d$label))

p_d <- ggplot(df_d, aes(x = prob, y = label, color = grp)) +
  geom_segment(aes(x = 0.948, xend = prob, y = label, yend = label), color = "#E2E8F0", linewidth = 0.7) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("Consistent InterPro signature" = "#4D908E", "De novo (no domain signature)" = "#D97A54")) +
  scale_x_continuous(limits = c(0.948, 1.008), breaks = seq(0.96, 1.00, 0.02)) +
  labs(x = "ZymeBind (Sequence only) probability", y = NULL, color = NULL) +
  theme(legend.position = c(0.30, 0.16), legend.key.size = unit(6.2, "pt"),
        legend.text = element_text(size = 5.1, family = "Arial"),
        axis.text.y = element_text(size = 5.2, family = "Arial", lineheight = 0.90))

# ---- Panels e & f: Pocket-Facing 3D Superpositions with Zoomed Active-Site Insets & Readable 7 pt Labels ----
p_e <- make_3d_panel(
  file.path(STRUCT_DIR, "plate_6_chitinase_superpos_tight.png"),
  "AF-A0AAD7U9D2 vs Chitinase 4MB5 (EC 3.2.1.14)\nFoldseek Prob = 1.000 | Sequence identity = 18.4%",
  "Steel-blue: AF-A0AAD7U9D2 | Surface: 4MB5\nInset: Nag5 substrate (orange) + DxDxE triad (cyan)"
)

p_f <- make_3d_panel(
  file.path(STRUCT_DIR, "plate_6_gst_superpos_tight.png"),
  "AF-A0AAD7ULU7 vs MGST2 6SSW (EC 2.5.1.18)\nFoldseek Prob = 1.000 | Sequence identity = 13.7%",
  "Slate-teal: AF-A0AAD7ULU7 | Surface: 6SSW\nInset: Glutathione (orange) + Gln34/Ser35 (cyan)"
)

# Use free(p_b) so p_b centers cleanly in column 2 without being pushed right by p_d's wide y-axis labels!
fig6 <- wrap_plots(p_a, free(p_b), p_c, p_d, p_e, p_f, ncol = 2, heights = c(1.02, 1.00, 1.40)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 232 / 25.4
out_p <- file.path(FIG_DIR, "Fig6_metagenome_dark_proteome")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(fig6); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(fig6); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(fig6); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(fig6); grDevices::dev.off()
cat("[EXPORTED] Fig6 (183 x 232 mm, Arial, centered rose, zoomed active-site pockets)\n")
