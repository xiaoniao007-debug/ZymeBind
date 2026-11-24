#!/usr/bin/env Rscript
# Supplementary Figure 1: Validation Set All-16 Model Benchmark, Single vs Multi-EC, Sequence Length Stratification & Blind-Test Calibration
# Strict Arial font, unified calm palette, unified nomenclature, no subpanel titles (a-d only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
OUT_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures/supplementary"

vs <- fromJSON(file.path(DATA_DIR, "val_sota_pk_complete.json"), simplifyVector = TRUE)
vb <- fromJSON(file.path(DATA_DIR, "val_baselines_strat_calib.json"), simplifyVector = TRUE)
bt <- fromJSON(file.path(DATA_DIR, "blind_test_common_clean_final_once.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"

# ---- Panel a: All 16 Models on Validation Set (n = 3,733) ----
m_map <- c(
  "knn_esm2"                = "ESM-2 650M kNN",
  "knn_dualplm"             = "Dual-PLM kNN",
  "sota_clean"              = "CLEAN",
  "mmseqs2_homology"        = "MMseqs2",
  "knn_prostt5"             = "ProstT5-3B kNN",
  "sota_enzymehunter"       = "EnzymeHunter",
  "sota_hitec"              = "HIT-EC",
  "sota_ecrecer"            = "ECRECer",
  "sota_deepec_transformer" = "DeepECtransformer",
  "sota_graphec_ld"         = "GraphEC-LD",
  "zb_single_seq_only"      = "ZymeBind (Single model, Sequence only)",
  "zb_ens4_seq_only"        = "ZymeBind (Sequence only)",
  "zb_single_seq_struct"    = "ZymeBind (Single model, Seq + Structure)",
  "zb_ens4_seq_struct"      = "ZymeBind (Seq + Structure)",
  "zb_single_full"          = "ZymeBind (Single model, All modalities)",
  "zb_ens4_full"            = "ZymeBind"
)
df_a <- do.call(rbind, lapply(names(m_map), function(k) {
  m <- vs$models[[k]]
  grp <- ifelse(k == "zb_ens4_full", "ZymeBind",
         ifelse(grepl("^zb_ens4", k), "ZymeBind (Biophysics)",
         ifelse(grepl("^zb_single", k), "ZymeBind (Single seed)", "Baseline")))
  data.frame(name = m_map[[k]], fmax = m$fmax, aupr = m$aupr_macro, grp = grp, stringsAsFactors = FALSE)
}))
df_a <- df_a[order(df_a$fmax), ]
df_a$name <- factor(df_a$name, levels = df_a$name)

p_a <- ggplot(df_a, aes(x = fmax, y = name, fill = grp)) +
  geom_col(width = 0.68, alpha = 0.94) +
  geom_point(aes(x = aupr), color = "#1E293B", size = 1.4, shape = 18) +
  scale_fill_manual(values = c("Baseline" = "#A9B4C2", "ZymeBind (Single seed)" = "#8FB6D9",
                               "ZymeBind (Biophysics)" = COL_TEAL, "ZymeBind" = COL_HERO)) +
  coord_cartesian(xlim = c(0.65, 1.00)) +
  scale_x_continuous(breaks = seq(0.70, 1.00, 0.10)) +
  labs(x = "Validation Fmax (bar) and AUPR-macro (diamond)", y = NULL, fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(5.5, "pt"),
        legend.text = element_text(size = 4.9, family = "Arial"),
        axis.text.y = element_text(size = 5.2, family = "Arial")) +
  guides(fill = guide_legend(nrow = 2))

# ---- Panel b: Single-EC vs Multi-EC Stratification ----
me <- vb$stratification$multi_ec
df_b <- rbind(
  data.frame(grp = "Single-EC\n(n = 3,625)", method = "MMseqs2",                  fmax = me$single_ec$mmseqs2$fmax,          mf1 = me$single_ec$mmseqs2$f1_macro_L4),
  data.frame(grp = "Single-EC\n(n = 3,625)", method = "ZymeBind (Sequence only)", fmax = me$single_ec$zb_ens4_seq_only$fmax, mf1 = me$single_ec$zb_ens4_seq_only$f1_macro_L4),
  data.frame(grp = "Single-EC\n(n = 3,625)", method = "ZymeBind",                 fmax = me$single_ec$zb_ens4_full$fmax,     mf1 = me$single_ec$zb_ens4_full$f1_macro_L4),
  data.frame(grp = "Multi-EC\n(n = 108)",    method = "MMseqs2",                  fmax = me$multi_ec$mmseqs2$fmax,           mf1 = me$multi_ec$mmseqs2$f1_macro_L4),
  data.frame(grp = "Multi-EC\n(n = 108)",    method = "ZymeBind (Sequence only)", fmax = me$multi_ec$zb_ens4_seq_only$fmax,  mf1 = me$multi_ec$zb_ens4_seq_only$f1_macro_L4),
  data.frame(grp = "Multi-EC\n(n = 108)",    method = "ZymeBind",                 fmax = me$multi_ec$zb_ens4_full$fmax,      mf1 = me$multi_ec$zb_ens4_full$f1_macro_L4)
)
df_b$grp <- factor(df_b$grp, levels = c("Single-EC\n(n = 3,625)", "Multi-EC\n(n = 108)"))
df_b$method <- factor(df_b$method, levels = c("MMseqs2", "ZymeBind (Sequence only)", "ZymeBind"))

p_b <- ggplot(df_b, aes(x = grp, y = fmax, fill = method)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  geom_text(aes(label = sprintf("%.3f", fmax)), position = position_dodge(width = 0.72),
            vjust = -0.35, size = 1.85, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = c("MMseqs2" = COL_SLATE, "ZymeBind (Sequence only)" = COL_SKY, "ZymeBind" = COL_HERO)) +
  coord_cartesian(ylim = c(0.75, 1.01)) +
  scale_y_continuous(breaks = seq(0.75, 1.00, 0.05)) +
  labs(x = "Enzyme functional multiplicity", y = "Fmax", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5.1, family = "Arial"))

# ---- Panel c: Sequence Length Stratification (<200 to >1,022 aa) ----
sl <- vb$stratification$seq_len
sl_keys <- c("lt200", "200_500", "500_1022", "gt1022")
sl_lbls <- c("<200 aa\n(n = 517)", "200-500 aa\n(n = 2,079)", "500-1,022 aa\n(n = 629)", ">1,022 aa\n(n = 508)")
df_c <- do.call(rbind, lapply(seq_along(sl_keys), function(i) {
  d <- sl[[sl_keys[i]]]
  rbind(
    data.frame(len = sl_lbls[i], method = "ZymeBind (Sequence only)", fmax = d$zb_ens4_seq_only),
    data.frame(len = sl_lbls[i], method = "ZymeBind",                 fmax = d$zb_ens4_full)
  )
}))
df_c$len <- factor(df_c$len, levels = sl_lbls)
df_c$method <- factor(df_c$method, levels = c("ZymeBind (Sequence only)", "ZymeBind"))

p_c <- ggplot(df_c, aes(x = len, y = fmax, color = method, group = method)) +
  geom_line(linewidth = 0.80) +
  geom_point(size = 2.3) +
  geom_text(aes(label = sprintf("%.3f", fmax), vjust = ifelse(method == "ZymeBind", -0.8, 1.5)),
            size = 1.85, family = "Arial", fontface = "bold", show.legend = FALSE) +
  scale_color_manual(values = c("ZymeBind (Sequence only)" = COL_TEAL, "ZymeBind" = COL_HERO)) +
  scale_y_continuous(limits = c(0.89, 1.01), breaks = seq(0.90, 1.00, 0.02)) +
  labs(x = "Protein sequence length stratum", y = "Fmax", color = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"),
        legend.text = element_text(size = 5.2, family = "Arial"),
        axis.text.x = element_text(size = 5.3, family = "Arial"))

# ---- Panel d: Independent Blind-Test Reliability Diagram (n = 2,841) ----
b_full <- bt$calibration$zb_ens4_full$reliability_bins
b_seq  <- bt$calibration$zb_ens4_seq_only$reliability_bins
df_d <- rbind(
  data.frame(conf = b_full$conf, acc = b_full$acc, n = b_full$n,
             model = sprintf("ZymeBind (Blind ECE = %.4f)", bt$calibration$zb_ens4_full$ece_temp_scaled_5fold)),
  data.frame(conf = b_seq$conf,  acc = b_seq$acc,  n = b_seq$n,
             model = sprintf("ZymeBind (Sequence only, ECE = %.4f)", bt$calibration$zb_ens4_seq_only$ece_temp_scaled_5fold))
) %>% filter(n >= 5)

p_d <- ggplot(df_d, aes(x = conf, y = acc, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.4) +
  geom_line(linewidth = 0.72) +
  geom_point(aes(size = log10(n + 1)), alpha = 0.9) +
  scale_color_manual(values = c(COL_HERO, COL_TEAL)) +
  scale_size_continuous(range = c(1.0, 2.6), guide = "none") +
  scale_x_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(x = "Predicted confidence (blind test)", y = "Observed top-1 accuracy", color = NULL) +
  theme(legend.position = "top", legend.direction = "vertical",
        legend.text = element_text(size = 5.2, family = "Arial"), legend.key.height = unit(7, "pt"))

sfig1 <- wrap_plots(p_a, free(p_b, type = "label"), free(p_c, type = "label"), free(p_d, type = "label"), ncol = 2) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(sfig1, file.path(OUT_DIR, "SuppFig1_val_strat_and_blind_calib"), width_mm = 183, height_mm = 148, panel_ids = c("a", "b", "c", "d"))
