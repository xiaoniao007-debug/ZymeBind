#!/usr/bin/env Rscript
# Figure 8: Mechanistic Dissection of Bifunctional & Zero-Homology Enzymes
# Strict Arial font, unified nomenclature (zero pid / DeepECt / tau* tags), zoomed active-site 3D plates + 7 pt labels, no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")
suppressPackageStartupMessages({
  library(png)
  library(grid)
})

DATA_DIR   <- "/Users/chenchen/Paper/ZymeBind/data"
STRUCT_DIR <- "/Users/chenchen/Paper/ZymeBind/structures"
FIG_DIR    <- "/Users/chenchen/Paper/ZymeBind/figures"

cs3 <- fromJSON(file.path(DATA_DIR, "case_study_3_hero_mechanistic.json"), simplifyVector = FALSE)
heroes <- cs3$top_10_dissected_heroes

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
      panel.border = element_rect(fill = NA, colour = "#CBD5E1", linewidth = 0.45),
      plot.margin = margin(25.9, 6, 4, 4, "pt")
    )
}

hero_names <- c(
  "B2TW38" = "B2TW38 (ArnA)\nBifunctional enzyme",
  "Q6UDJ5" = "Q6UDJ5 (PsHV-1)\n24.7% identity",
  "Q9HPK9" = "Q9HPK9 (H. salinarum)\n0% identity (no hit)",
  "Q8V3U0" = "Q8V3U0 (ISAV)\n0% identity (no hit)"
)

# ---- Panel a: 6-Model Side-by-Side Prediction Matrix ----
model_cols <- c("sota_clean", "sota_enzymehunter", "sota_deepec_transformer", "sota_hitec", "sota_graphec_ld", "zb")
model_labs <- c("CLEAN", "Enzyme-\nHunter", "DeepEC-\ntransformer", "HIT-EC", "GraphEC-\nLD", "ZymeBind")

rows_a <- list()
for (h in heroes) {
  acc <- h$accession
  true_ecs <- unlist(h$true_ecs)
  for (mi in seq_along(model_cols)) {
    mk <- model_cols[mi]
    if (mk == "zb") {
      pred_ec <- h$zb_seq_struct_top1; prob <- h$zb_seq_struct_prob
    } else {
      pred_ec <- h$sota_top1[[mk]][[1]]; prob <- as.numeric(h$sota_top1[[mk]][[2]])
    }
    rows_a[[length(rows_a) + 1]] <- data.frame(
      hero = hero_names[[acc]], model = model_labs[mi],
      status = ifelse(pred_ec %in% true_ecs, "Correct EC4", "Misclassified"),
      cell_txt = sprintf("%s\n(%.2f)", pred_ec, prob), stringsAsFactors = FALSE
    )
  }
}
df_a <- do.call(rbind, rows_a)
df_a$hero  <- factor(df_a$hero, levels = rev(unname(hero_names)))
df_a$model <- factor(df_a$model, levels = model_labs)

p_a <- ggplot(df_a, aes(x = model, y = hero, fill = status)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = cell_txt), size = 1.78, family = "Arial", color = "#111111", lineheight = 0.9) +
  scale_fill_manual(values = c("Correct EC4" = "#D9EBE9", "Misclassified" = "#F4DDD2")) +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "top", legend.key.size = unit(7, "pt"),
        axis.text.x = element_text(size = 5.2, family = "Arial", face = "bold"),
        axis.text.y = element_text(size = 5.4, family = "Arial", lineheight = 0.92))

# ---- Panel b: Per-Enzyme Exact 4-Modality Shapley Attribution ----
rows_b <- list()
for (h in heroes) {
  sh <- h$l2_exact_shapley$shapley
  rows_b[[length(rows_b) + 1]] <- data.frame(
    hero = hero_names[[h$accession]],
    modality = factor(c("Sequence", "Structure", "Text", "Domain"), levels = c("Domain", "Text", "Structure", "Sequence")),
    val = c(sh$sequence, sh$structure, sh$text, sh$domain), stringsAsFactors = FALSE
  )
}
df_b <- do.call(rbind, rows_b)
df_b$hero <- factor(df_b$hero, levels = rev(unname(hero_names)))

p_b <- ggplot(df_b, aes(x = val, y = hero, fill = modality)) +
  geom_col(width = 0.60, color = "white", linewidth = 0.35) +
  geom_text(aes(label = ifelse(val >= 0.14, sprintf("%.2f", val), "")),
            position = position_stack(vjust = 0.5), size = 1.9, family = "Arial", color = "white", fontface = "bold") +
  scale_fill_manual(values = PAL_MODALITY) +
  scale_x_continuous(limits = c(0, 1.05), expand = c(0, 0), breaks = seq(0, 1.0, 0.25)) +
  labs(x = "Shapley contribution to true EC4 probability", y = NULL, fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.5, "pt"),
        axis.text.y = element_text(size = 5.4, family = "Arial", lineheight = 0.92))

# ---- Panel c: Modality-Robust Confidence ----
rows_c <- list()
for (h in heroes) {
  rows_c[[length(rows_c) + 1]] <- rbind(
    data.frame(hero = h$accession, mode = "ZymeBind (Sequence only)",   prob = h$zb_seq_only_prob),
    data.frame(hero = h$accession, mode = "ZymeBind (Seq + Structure)", prob = h$zb_seq_struct_prob),
    data.frame(hero = h$accession, mode = "ZymeBind",                   prob = h$l2_exact_shapley$v_full)
  )
}
df_c <- do.call(rbind, rows_c)
df_c$hero <- factor(df_c$hero, levels = c("B2TW38", "Q6UDJ5", "Q9HPK9", "Q8V3U0"))
df_c$mode <- factor(df_c$mode, levels = c("ZymeBind (Sequence only)", "ZymeBind (Seq + Structure)", "ZymeBind"))

p_c <- ggplot(df_c, aes(x = hero, y = prob, fill = mode)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  geom_hline(yintercept = 0.62, linetype = "dashed", color = "#64748B", linewidth = 0.38) +
  scale_fill_manual(values = c("ZymeBind (Sequence only)" = "#6B9AC4", "ZymeBind (Seq + Structure)" = "#4D908E", "ZymeBind" = "#2B5C8F")) +
  scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0), breaks = seq(0, 1.0, 0.2)) +
  labs(x = NULL, y = "Predicted true-EC4 probability", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"), legend.text = element_text(size = 5.0, family = "Arial"))

# ---- Panel d: Catalytic Active-Site Sequence Map ----
df_len <- data.frame(
  hero = c("B2TW38 (660 aa)\nEC 1.1.1.305 / 2.1.2.13", "Q6UDJ5 (238 aa)\nEC 3.4.19.12", "Q9HPK9 (231 aa)\nEC 3.5.1.90", "Q8V3U0 (388 aa)\nEC 3.1.1.53"),
  len  = c(660, 238, 231, 388), stringsAsFactors = FALSE
)
df_len$hero <- factor(df_len$hero, levels = rev(df_len$hero))

df_sites <- data.frame(
  hero = factor(c(
    "B2TW38 (660 aa)\nEC 1.1.1.305 / 2.1.2.13", "B2TW38 (660 aa)\nEC 1.1.1.305 / 2.1.2.13",
    "Q6UDJ5 (238 aa)\nEC 3.4.19.12", "Q6UDJ5 (238 aa)\nEC 3.4.19.12", "Q6UDJ5 (238 aa)\nEC 3.4.19.12",
    "Q9HPK9 (231 aa)\nEC 3.5.1.90", "Q9HPK9 (231 aa)\nEC 3.5.1.90",
    "Q8V3U0 (388 aa)\nEC 3.1.1.53", "Q8V3U0 (388 aa)\nEC 3.1.1.53", "Q8V3U0 (388 aa)\nEC 3.1.1.53"
  ), levels = levels(df_len$hero)),
  pos = c(300, 485, 30, 173, 175, 79, 164, 32, 261, 264),
  lab = c("His300", "Tyr485", "Cys30", "His173", "Asp175", "His79", "Asp164", "Ser32", "Asp261", "His264"),
  stringsAsFactors = FALSE
)

p_d <- ggplot(df_len, aes(y = hero)) +
  geom_segment(aes(x = 1, xend = len, yend = hero), color = "#CBD5E1", linewidth = 4.2, lineend = "round") +
  geom_point(data = df_sites, aes(x = pos, y = hero), color = "#D97A54", size = 2.3) +
  geom_text_repel(data = df_sites, aes(x = pos, y = hero, label = lab),
                  size = 1.85, family = "Arial", color = "#B85D3B", fontface = "bold", nudge_y = 0.28,
                  min.segment.length = 0.1, segment.size = 0.25, seed = 42) +
  scale_x_continuous(limits = c(0, 700), breaks = seq(0, 600, 150)) +
  labs(x = "Residue position (amino acids)", y = NULL) +
  theme(axis.text.y = element_text(size = 5.3, family = "Arial", lineheight = 0.92))

# ---- Panels e & f: Pocket-Facing 3D Structures with Zoomed Catalytic-Site Insets & 7 pt Labels ----
p_e <- make_3d_panel(
  file.path(STRUCT_DIR, "plate_8_B2TW38_arnA_tight.png"),
  "Bifunctional ArnA (B2TW38, EC 1.1.1.305 / 2.1.2.13)\nPredicted P = 0.996 | AlphaFold mean pLDDT = 93.4",
  "Steel-blue: N-domain | Slate-teal: C-domain\nInset: Catalytic His300 (orange) + Tyr485 (cyan)"
)

p_f <- make_3d_panel(
  file.path(STRUCT_DIR, "plate_8_Q9HPK9_archaeon_tight.png"),
  "Archaeal CbiZ (Q9HPK9, EC 3.5.1.90, 0% identity)\nPredicted P = 0.967 | AlphaFold mean pLDDT = 93.1",
  "Blue gradient: pLDDT confidence\nInset: Conserved His50/His79/Asp164/Glu168 cleft (cyan)"
)

fig8 <- wrap_plots(p_a, p_b, free(p_c, type = "label"), p_d, p_e, p_f, ncol = 2, heights = c(1, 1, 1.42)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(
  fig8,
  file.path(FIG_DIR, "Fig8_hero_mechanistic_dissection"),
  width_mm = 183, height_mm = 228,
  panel_ids = c("a", "b", "c", "d", "e", "f")
)
