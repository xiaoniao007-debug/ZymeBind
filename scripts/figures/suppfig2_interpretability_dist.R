#!/usr/bin/env Rscript
# Supplementary Figure 2: Deep Interpretability Distributions (L1 Per-Protein ECDF, L2 EC1 Shapley Shifts & L3 SAE Top-10 Concepts)
# Strict Arial font, unified calm palette, no subpanel titles (a-c only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
OUT_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures/supplementary"

l1  <- fromJSON(file.path(DATA_DIR, "interp_l1_site_masking_v2_p1m1_test.json"), simplifyVector = TRUE)
sh2 <- fromJSON(file.path(DATA_DIR, "v2_interpretability_shapley_l2_p1m1_test.json"), simplifyVector = TRUE)
l3  <- fromJSON(file.path(DATA_DIR, "val_interpretability_l3_sae_monosemantic.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"

# ---- Panel a: L1 Per-Protein Paired Difference ECDF (Site Drop minus Random Control Drop, n = 400) ----
pa <- l1$per_protein$act_site
pb <- l1$per_protein$act_or_bind
df_a <- rbind(
  data.frame(cond = "Catalytic site (n = 400)",             diff = (pa$drop_site - pa$drop_rand_mean) * 1000),
  data.frame(cond = "Catalytic + binding (n = 400)",        diff = (pb$drop_site - pb$drop_rand_mean) * 1000),
  data.frame(cond = "Catalytic + binding (correct, n = 244)", diff = (pb$drop_site[pb$correct] - pb$drop_rand_mean[pb$correct]) * 1000)
)
df_a$cond <- factor(df_a$cond, levels = c(
  "Catalytic site (n = 400)", "Catalytic + binding (n = 400)", "Catalytic + binding (correct, n = 244)"
))
df_a$diff_clip <- pmin(pmax(df_a$diff, -10), 25)

p_a <- ggplot(df_a, aes(x = diff_clip, color = cond)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#64748B", linewidth = 0.4) +
  stat_ecdf(linewidth = 0.78) +
  scale_color_manual(values = c("Catalytic site (n = 400)" = COL_SKY,
                                "Catalytic + binding (n = 400)" = COL_TEAL,
                                "Catalytic + binding (correct, n = 244)" = COL_HERO)) +
  scale_x_continuous(limits = c(-10, 25), breaks = seq(-10, 25, 5)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.25)) +
  labs(x = "Paired probability drop difference (site - random, x 10^-3)", y = "Empirical cumulative fraction", color = NULL) +
  theme(legend.position = c(0.63, 0.24), legend.text = element_text(size = 5.1, family = "Arial"),
        legend.key.height = unit(7, "pt"))

# ---- Panel b: L2 Exact Shapley Biophysical Share (Sequence + Structure %) Across All 7 EC1 Classes ----
ec1_keys <- names(sh2$v1$by_ec1)
ec1_short <- sub("_.*", "", ec1_keys)
df_b <- do.call(rbind, lapply(seq_along(ec1_keys), function(i) {
  k <- ec1_keys[i]
  v1s <- sh2$v1$by_ec1[[k]]$shapley_share_pct
  v2s <- sh2$v2$by_ec1[[k]]$shapley_share_pct
  rbind(
    data.frame(ec1 = ec1_short[i], model = "Standard fusion",      biophys = v1s$sequence + v1s$structure),
    data.frame(ec1 = ec1_short[i], model = "+ Modality dropout",   biophys = v2s$sequence + v2s$structure)
  )
}))
df_b$ec1 <- factor(df_b$ec1, levels = ec1_short)
df_b$model <- factor(df_b$model, levels = c("Standard fusion", "+ Modality dropout"))

p_b <- ggplot(df_b, aes(x = ec1, y = biophys, fill = model)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  geom_text(aes(label = sprintf("%.0f%%", biophys)), position = position_dodge(width = 0.72),
            vjust = -0.35, size = 1.80, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = c("Standard fusion" = "#CBD5E1", "+ Modality dropout" = COL_HERO)) +
  scale_y_continuous(limits = c(0, 68), expand = c(0, 0), breaks = seq(0, 60, 15)) +
  labs(x = "Primary EC1 enzyme class", y = "Biophysical Shapley share (Seq + Struct, %)", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"),
        legend.text = element_text(size = 5.3, family = "Arial"))

# ---- Panel c: L3 Top-K SAE Full Top-10 Single-Neuron Alignment Across All 3 Concept Families (30 concepts) ----
get_t10 <- function(tbl, cat_lbl) {
  data.frame(
    name = sprintf("%s (#%d)", tbl$concept, tbl$feature_id),
    f1 = tbl$f1, auroc = tbl$auroc, n_pos = tbl$n_pos, cat = cat_lbl, stringsAsFactors = FALSE
  )
}
df_c <- rbind(
  get_t10(l3$interpro_domains$top10, "InterPro domain"),
  get_t10(l3$ec3_mechanisms$top10,   "EC3 sub-subclass"),
  get_t10(l3$cofactors$top10,        "Catalytic cofactor")
)
df_c$cat <- factor(df_c$cat, levels = c("InterPro domain", "EC3 sub-subclass", "Catalytic cofactor"))

p_c <- ggplot(df_c, aes(x = f1, y = auroc, color = cat, size = log10(n_pos + 1))) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "#CBD5E1", linewidth = 0.4) +
  geom_point(alpha = 0.85) +
  ggrepel::geom_text_repel(
    data = df_c %>% filter(cat == "Catalytic cofactor" | f1 < 0.90 | grepl("^5\\.2\\.1|^IPR029028", name)),
    aes(label = name), size = 1.65, family = "Arial", max.overlaps = 14, box.padding = 0.20,
    segment.size = 0.22, show.legend = FALSE, seed = 42
  ) +
  scale_color_manual(values = c("InterPro domain" = COL_HERO, "EC3 sub-subclass" = COL_TEAL, "Catalytic cofactor" = COL_TERRA)) +
  scale_size_continuous(range = c(1.5, 3.2), guide = "none") +
  scale_x_continuous(limits = c(0.15, 1.03), breaks = seq(0.2, 1.0, 0.2)) +
  scale_y_continuous(limits = c(0.45, 1.03), breaks = seq(0.5, 1.0, 0.1)) +
  labs(x = "Single-neuron F1 score", y = "Single-neuron AUROC", color = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5.2, family = "Arial"))

top_r <- (p_a | p_b) + plot_layout(widths = c(1.0, 1.05))
sfig2 <- (top_r / p_c) +
  plot_layout(heights = c(1.0, 1.05)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 150 / 25.4
out_p <- file.path(OUT_DIR, "SuppFig2_interpretability_full_distributions")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(sfig2); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(sfig2); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(sfig2); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(sfig2); grDevices::dev.off()
cat("[EXPORTED] SuppFig2 (183 x 150 mm, Arial)\n")
