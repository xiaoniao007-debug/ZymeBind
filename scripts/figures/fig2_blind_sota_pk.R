#!/usr/bin/env Rscript
# Figure 2: Independent Blind Test Evaluation (n = 2,841, 673 clusters)
# Strict Arial font, unified nomenclature, no subpanel titles (a-e only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
FIG_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures"

bt     <- fromJSON(file.path(DATA_DIR, "blind_test_common_clean_final_once.json"), simplifyVector = TRUE)
ecr    <- fromJSON(file.path(DATA_DIR, "blind_test_ecrecer_addendum_final_once.json"), simplifyVector = TRUE)
val_bs <- fromJSON(file.path(DATA_DIR, "val_baselines_strat_calib.json"), simplifyVector = TRUE)
bt$models$sota_ecrecer <- ecr$common_clean_test

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_ICE   <- "#C6D9EC"
COL_SLATE <- "#8D99AE"

# ---- Panel a: EC1 Class Composition Donut Chart (enlarged & shifted left) ----
ec1_raw <- val_bs$stratification$ec1
ec1_keys <- paste0("EC", 1:7)
df_donut <- data.frame(
  ec1 = ec1_keys,
  n   = sapply(ec1_keys, function(k) as.numeric(ec1_raw[[k]]$n)),
  stringsAsFactors = FALSE
)
df_donut$frac <- df_donut$n / sum(df_donut$n)
df_donut$ymax <- cumsum(df_donut$frac)
df_donut$ymin <- c(0, head(df_donut$ymax, -1))
df_donut$ymid <- 0.5 * (df_donut$ymin + df_donut$ymax)
df_donut$lbl  <- ifelse(df_donut$frac >= 0.06, sprintf("%s\n%.0f%%", df_donut$ec1, 100 * df_donut$frac), "")
donut_cols <- c("EC1" = "#2B5C8F", "EC2" = "#5B8DBE", "EC3" = "#4D908E",
                "EC4" = "#8FB6D9", "EC5" = "#D97A54", "EC6" = "#B0BAC9", "EC7" = "#E5B89B")

p_a <- ggplot(df_donut, aes(ymax = ymax, ymin = ymin, xmax = 4.00, xmin = 2.15, fill = ec1)) +
  geom_rect(color = "#FFFFFF", linewidth = 0.55) +
  geom_text(aes(x = 3.10, y = ymid, label = lbl), size = 1.95, family = "Arial", fontface = "bold", color = "#FFFFFF", lineheight = 0.86) +
  annotate("text", x = 0.65, y = 0, label = "673 clusters\nn = 2,841", size = 2.15, family = "Arial", fontface = "bold", color = "#1E293B", lineheight = 1.05) +
  coord_polar(theta = "y") +
  xlim(c(0.65, 4.04)) +
  scale_fill_manual(values = donut_cols) +
  labs(fill = NULL) +
  theme_void(base_family = "Arial") +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 5.4, family = "Arial"),
    legend.key.size = unit(6.0, "pt"),
    plot.margin = margin(-4, 6, 0, 2, "pt"),
    plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
  ) +
  guides(fill = guide_legend(nrow = 2))

# ---- Panel b: 6-Axis Polar Multi-Metric Radar (enlarged & thinner crisp strokes) ----
radar_models <- c(
  "zb_ens4_full"            = "ZymeBind",
  "zb_ens4_seq_only"        = "ZymeBind (Sequence only)",
  "sota_graphec_ld"         = "GraphEC-LD",
  "sota_deepec_transformer" = "DeepECtransformer",
  "sota_clean"              = "CLEAN",
  "mmseqs2_homology"        = "MMseqs2"
)
axes_lbl <- c("Fmax", "AUPR-macro", "Weighted-F1", "Macro-F1", "Rare EC4", "Identity <20%")
n_ax <- length(axes_lbl)
angles <- seq(pi / 2, pi / 2 - 2 * pi, length.out = n_ax + 1)[1:n_ax]

df_rad <- do.call(rbind, lapply(names(radar_models), function(k) {
  m <- bt$models[[k]]
  twi <- bt$homology_decay_curve$lt20$fmax[[k]]
  raw <- c(m$fmax, m$aupr_macro, m$weighted_f1, m$macro_f1_L4, m$rare_fmax, twi)
  r_val <- 0.15 + 0.85 * raw
  ang_c <- c(angles, angles[1]); r_c <- c(r_val, r_val[1])
  data.frame(model = radar_models[[k]], x = r_c * cos(ang_c), y = r_c * sin(ang_c), stringsAsFactors = FALSE)
}))
df_rad$model <- factor(df_rad$model, levels = unname(radar_models))
df_rad$is_main <- df_rad$model == "ZymeBind"

grid_rings <- do.call(rbind, lapply(c(0.25, 0.50, 0.75, 1.00), function(val) {
  rr <- 0.15 + 0.85 * val; ang_c <- c(angles, angles[1])
  data.frame(ring = val, x = rr * cos(ang_c), y = rr * sin(ang_c))
}))
spokes <- data.frame(
  x = 0.15 * cos(angles), y = 0.15 * sin(angles),
  xend = 1.0 * cos(angles), yend = 1.0 * sin(angles),
  lx = 1.22 * cos(angles), ly = 1.13 * sin(angles), lbl = axes_lbl
)
rad_cols <- c("ZymeBind" = COL_HERO, "ZymeBind (Sequence only)" = COL_TEAL, "GraphEC-LD" = COL_SKY,
              "DeepECtransformer" = COL_TERRA, "CLEAN" = "#9EB3C2", "MMseqs2" = COL_SLATE)

p_b <- ggplot() +
  geom_path(data = grid_rings, aes(x = x, y = y, group = ring), color = "#E2E8F0", linewidth = 0.30) +
  geom_segment(data = spokes, aes(x = x, y = y, xend = xend, yend = yend), color = "#CBD5E1", linewidth = 0.26) +
  geom_polygon(data = df_rad, aes(x = x, y = y, color = model, fill = model, group = model, linewidth = is_main), alpha = 0.03) +
  geom_point(data = df_rad, aes(x = x, y = y, color = model), size = 0.65) +
  geom_text(data = spokes, aes(x = lx, y = ly, label = lbl), size = 1.90, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_color_manual(values = rad_cols) +
  scale_fill_manual(values = rad_cols, guide = "none") +
  scale_linewidth_manual(values = c("FALSE" = 0.30, "TRUE" = 0.56), guide = "none") +
  coord_equal(xlim = c(-1.32, 1.32), ylim = c(-1.16, 1.16), clip = "off") +
  labs(color = NULL) +
  theme_void(base_family = "Arial") +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 5.0, family = "Arial"),
    legend.key.size = unit(5.8, "pt"),
    plot.margin = margin(0, 4, 0, -4, "pt"),
    plot.tag = element_text(size = 8.5, family = "Arial", face = "bold")
  ) +
  guides(color = guide_legend(nrow = 3, override.aes = list(fill = NA, linewidth = 0.8)))

# ---- Panel c: Sequence-Identity Decay Curve (thinner separated trajectories) ----
bins <- c("lt20", "20_30", "30_40", "ge40")
bin_labels <- c("<20%\n(62)", "20-30%\n(82)", "30-40%\n(1,004)", ">=40%\n(1,693)")
curve_models <- c(
  "zb_ens4_full"            = "ZymeBind",
  "zb_ens4_seq_struct"      = "ZymeBind (Seq + Structure)",
  "zb_ens4_seq_only"        = "ZymeBind (Sequence only)",
  "sota_graphec_ld"         = "GraphEC-LD",
  "sota_deepec_transformer" = "DeepECtransformer",
  "sota_clean"              = "CLEAN",
  "sota_hitec"              = "HIT-EC",
  "mmseqs2_homology"        = "MMseqs2"
)
df_c <- do.call(rbind, lapply(names(curve_models), function(k) {
  data.frame(model = curve_models[[k]], bin_num = 1:4,
             fmax = sapply(bins, function(b) bt$homology_decay_curve[[b]]$fmax[[k]]), stringsAsFactors = FALSE)
}))
df_c$model <- factor(df_c$model, levels = unname(curve_models))
df_c$is_zb <- grepl("^ZymeBind", df_c$model)
df_rib <- data.frame(
  bin_num = 1:4,
  ymin = sapply(bins, function(b) bt$homology_decay_curve[[b]]$fmax$mmseqs2_homology),
  ymax = sapply(bins, function(b) bt$homology_decay_curve[[b]]$fmax$zb_ens4_full)
)
pal_c <- c("ZymeBind" = COL_HERO, "ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind (Sequence only)" = COL_SKY,
           "GraphEC-LD" = "#8FB6D9", "DeepECtransformer" = COL_TERRA, "CLEAN" = "#A3B8C8", "HIT-EC" = "#B8A9C9", "MMseqs2" = COL_SLATE)

p_c <- ggplot() +
  geom_ribbon(data = df_rib, aes(x = bin_num, ymin = ymin, ymax = ymax), fill = COL_ICE, alpha = 0.38) +
  geom_line(data = df_c, aes(x = bin_num, y = fmax, color = model, group = model, linewidth = is_zb, linetype = model == "MMseqs2")) +
  geom_point(data = df_c, aes(x = bin_num, y = fmax, color = model, size = is_zb)) +
  scale_color_manual(values = pal_c) +
  scale_linewidth_manual(values = c("FALSE" = 0.34, "TRUE" = 0.62), guide = "none") +
  scale_size_manual(values = c("FALSE" = 0.78, "TRUE" = 1.25), guide = "none") +
  scale_linetype_manual(values = c("FALSE" = "solid", "TRUE" = "dashed"), guide = "none") +
  scale_x_continuous(breaks = 1:4, labels = bin_labels, expand = expansion(mult = c(0.08, 0.08))) +
  scale_y_continuous(limits = c(-0.04, 1.05), breaks = seq(0, 1, 0.25)) +
  annotate("text", x = 1.03, y = 0.75, label = "+0.578 vs\nMMseqs2", size = 1.85, family = "Arial", fontface = "bold", color = COL_HERO, hjust = 0, lineheight = 0.92) +
  labs(x = "Maximum sequence identity to training set", y = "Fmax", color = NULL) +
  theme(legend.position = c(0.68, 0.32), legend.key.height = unit(5.8, "pt"),
        legend.text = element_text(size = 4.9, family = "Arial"), axis.text.x = element_text(size = 5.4, family = "Arial"))

# ---- Panel d: 40%-Cluster Paired Bootstrap Forest Plot ----
boot_list <- bt$significance_bootstrap_40pct_cluster
boot_list$zb_ens4_full_minus_sota_ecrecer <- ecr$paired_bootstrap_vs_ecrecer$zb_ens4_full_minus_sota_ecrecer
pairs_order <- c(
  "zb_ens4_full_minus_mmseqs2_homology"        = "ZymeBind vs MMseqs2",
  "zb_ens4_full_minus_sota_clean"              = "ZymeBind vs CLEAN",
  "zb_ens4_full_minus_sota_enzymehunter"       = "ZymeBind vs EnzymeHunter",
  "zb_ens4_full_minus_sota_ecrecer"            = "ZymeBind vs ECRECer",
  "zb_ens4_full_minus_sota_deepec_transformer" = "ZymeBind vs DeepECtransformer",
  "zb_ens4_full_minus_sota_graphec_ld"         = "ZymeBind vs GraphEC-LD",
  "zb_ens4_seq_only_minus_sota_hitec"          = "ZymeBind (Sequence only) vs HIT-EC",
  "zb_ens4_seq_only_minus_mmseqs2_homology"    = "ZymeBind (Sequence only) vs MMseqs2",
  "zb_ens4_seq_only_minus_sota_clean"          = "ZymeBind (Sequence only) vs CLEAN",
  "zb_ens4_seq_struct_minus_sota_enzymehunter" = "ZymeBind (Seq + Structure) vs EnzymeHunter",
  "zb_ens4_seq_struct_minus_sota_graphec_ld"   = "ZymeBind (Seq + Structure) vs GraphEC-LD"
)
df_d <- do.call(rbind, lapply(names(pairs_order), function(k) {
  x <- boot_list[[k]]; if (is.null(x)) return(NULL)
  data.frame(pair = pairs_order[[k]], delta = x$diff, lo = x$ci95[1], hi = x$ci95[2], pval = x$p_value,
             grp = ifelse(grepl("^ZymeBind vs", pairs_order[[k]]), "ZymeBind (All modalities)", "Biophysics only"), stringsAsFactors = FALSE)
}))
df_d$pair <- factor(df_d$pair, levels = rev(unique(df_d$pair)))

p_d <- ggplot(df_d, aes(x = delta, y = pair, color = grp)) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#EEF5FB", alpha = 0.65) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#64748B", linewidth = 0.38) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.28, linewidth = 0.58, orientation = "y") +
  geom_point(size = 1.9) +
  geom_text(aes(x = hi + 0.003, label = sprintf("p = %.4f", pval)), hjust = 0, size = 1.75, family = "Arial", show.legend = FALSE) +
  scale_color_manual(values = c("ZymeBind (All modalities)" = COL_HERO, "Biophysics only" = COL_TEAL)) +
  scale_x_continuous(limits = c(-0.012, 0.128), breaks = seq(0, 0.10, 0.03)) +
  labs(x = "Paired F1 difference at locked threshold (95% CI)", y = NULL, color = NULL) +
  theme(legend.position = c(0.68, 0.15), legend.key.height = unit(7, "pt"),
        legend.text = element_text(size = 5.2, family = "Arial"),
        axis.text.y = element_text(size = 5.4, family = "Arial"))

# ---- Panel e: 2D Pareto Frontier + Embedded Overall Fmax Inset ----
models_all <- c(
  "zb_ens4_full"            = "ZymeBind",
  "zb_ens4_seq_struct"      = "ZymeBind (Seq + Structure)",
  "zb_ens4_seq_only"        = "ZymeBind (Sequence only)",
  "sota_graphec_ld"         = "GraphEC-LD",
  "sota_deepec_transformer" = "DeepECtransformer",
  "sota_enzymehunter"       = "EnzymeHunter",
  "sota_ecrecer"            = "ECRECer",
  "sota_clean"              = "CLEAN",
  "sota_hitec"              = "HIT-EC",
  "knn_dualplm"             = "Dual-PLM kNN",
  "mmseqs2_homology"        = "MMseqs2"
)
df_e <- do.call(rbind, lapply(names(models_all), function(k) {
  m <- bt$models[[k]]
  data.frame(name = models_all[[k]], mf1 = m$macro_f1_L4, rare = m$rare_fmax,
             is_zb = grepl("^ZymeBind", models_all[[k]]), stringsAsFactors = FALSE)
}))
gx <- seq(0.65, 0.95, length.out = 60); gy <- seq(0.16, 0.97, length.out = 60)
grid_iso <- expand.grid(x = gx, y = gy)
grid_iso$h <- 2 * grid_iso$x * grid_iso$y / (grid_iso$x + grid_iso$y)

pal_e <- c("ZymeBind" = COL_HERO, "ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind (Sequence only)" = COL_SKY,
           "GraphEC-LD" = "#7B9EC2", "DeepECtransformer" = COL_TERRA, "EnzymeHunter" = "#C89D8B",
           "ECRECer" = "#88A9A4", "CLEAN" = "#9EB3C2", "HIT-EC" = "#A99EBA", "Dual-PLM kNN" = "#BCC4CE", "MMseqs2" = COL_SLATE)

df_ins <- data.frame(
  m = factor(c("MMseqs2", "CLEAN", "GraphEC-LD", "ZymeBind"), levels = c("MMseqs2", "CLEAN", "GraphEC-LD", "ZymeBind")),
  v = c(bt$models$mmseqs2_homology$fmax, bt$models$sota_clean$fmax, bt$models$sota_graphec_ld$fmax, bt$models$zb_ens4_full$fmax),
  col = c(COL_SLATE, "#9EB3C2", "#7B9EC2", COL_HERO)
)
p_ins <- ggplot(df_ins, aes(x = m, y = v, fill = col)) +
  geom_col(width = 0.62) +
  scale_fill_identity() +
  coord_cartesian(ylim = c(0.86, 0.99)) +
  scale_y_continuous(breaks = c(0.88, 0.93, 0.98)) +
  labs(x = NULL, y = "Fmax") +
  theme_nature() +
  theme(
    axis.title.y = element_text(size = 5.0, family = "Arial"),
    axis.text.x = element_text(size = 4.4, family = "Arial", angle = 22, hjust = 1),
    axis.text.y = element_text(size = 4.5, family = "Arial"),
    plot.background = element_rect(fill = "#FFFFFF", color = "#CBD5E1", linewidth = 0.35),
    plot.margin = margin(3, 3, 2, 2, "pt")
  )

tmp_ins <- tempfile(fileext = ".png")
ragg::agg_png(tmp_ins, width = 2, height = 1.5, units = "in", res = 150)
ins_grob <- ggplotGrob(p_ins)
grDevices::dev.off(); unlink(tmp_ins)

p_e <- ggplot() +
  geom_contour(data = grid_iso, aes(x = x, y = y, z = h),
               breaks = c(0.45, 0.60, 0.75, 0.85, 0.90), color = "#CBD5E1", linetype = "dotted", linewidth = 0.42) +
  annotate("rect", xmin = 0.825, xmax = 0.952, ymin = 0.77, ymax = 0.97, fill = COL_ICE, alpha = 0.32) +
  annotate("text", x = 0.855, y = 0.955, label = "H = 0.90", size = 1.55, family = "Arial", color = "#64748B", hjust = 0.5) +
  annotate("text", x = 0.949, y = 0.635, label = "H = 0.75", size = 1.55, family = "Arial", color = "#64748B", hjust = 1) +
  geom_point(data = df_e, aes(x = mf1, y = rare, color = name, size = is_zb), alpha = 0.95) +
  ggrepel::geom_text_repel(
    data = df_e, aes(x = mf1, y = rare, label = name, color = name, fontface = ifelse(is_zb, "bold", "plain")),
    size = 1.82, family = "Arial", box.padding = 0.24, point.padding = 0.18, segment.size = 0.24, max.overlaps = 20, show.legend = FALSE, seed = 42
  ) +
  annotation_custom(ins_grob, xmin = 0.800, xmax = 0.953, ymin = 0.165, ymax = 0.435) +
  scale_color_manual(values = pal_e, guide = "none") +
  scale_size_manual(values = c("FALSE" = 2.2, "TRUE" = 3.3), guide = "none") +
  scale_x_continuous(limits = c(0.65, 0.956), breaks = seq(0.68, 0.92, 0.06)) +
  scale_y_continuous(limits = c(0.16, 0.98), breaks = seq(0.20, 0.95, 0.15)) +
  labs(x = "Macro-F1 across EC4 classes (locked threshold)", y = "Rare EC4 Fmax (1-5 training examples)")

# Decouple top_row from bot_row's wide y-axis margin via wrap_elements so Panel a sits flush left!
top_row <- (p_a | p_b | p_c) + plot_layout(widths = c(0.95, 1.08, 1.15))
bot_row <- (p_d | p_e) + plot_layout(widths = c(1.06, 1.04))
fig2 <- (free(top_row) / bot_row) +
  plot_layout(heights = c(1.02, 1.14)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 156 / 25.4
out_p <- file.path(FIG_DIR, "Fig2_blind_sota_pk")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(fig2); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(fig2); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(fig2); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(fig2); grDevices::dev.off()
cat("[EXPORTED] Fig2 (183 x 156 mm, Arial, unified nomenclature)\n")
