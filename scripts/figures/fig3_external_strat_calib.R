#!/usr/bin/env Rscript
# Figure 3: External Benchmarks (NEW-392 & Price-149), EC1 Stratification & Probability Calibration
# Strict Arial font, unified nomenclature (zero Frozen-2022 / obs3Di / ZymeBind-Single dev tags), no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
FIG_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures"

ext    <- fromJSON(file.path(DATA_DIR, "blind_test_external_new392_price149_final_once.json"), simplifyVector = TRUE)
ecr    <- fromJSON(file.path(DATA_DIR, "blind_test_ecrecer_addendum_final_once.json"), simplifyVector = TRUE)
val_bs <- fromJSON(file.path(DATA_DIR, "val_baselines_strat_calib.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"

# ---- Panel a: NEW-392 External Benchmark ----
n392_sota <- ext$benchmarks$new392$sota_same_split
n392_sota$sota_ecrecer <- ecr$external_benchmarks$new392
n392_zb <- ext$benchmarks$new392$zymebind_locked

df_a <- rbind(
  data.frame(name = "HIT-EC",                      fmax = n392_sota$sota_hitec$fmax,              wf1 = n392_sota$sota_hitec$weighted_f1,              grp = "Baseline"),
  data.frame(name = "GraphEC-LD",                  fmax = n392_sota$sota_graphec_ld$fmax,         wf1 = n392_sota$sota_graphec_ld$weighted_f1,         grp = "Baseline"),
  data.frame(name = "CLEAN",                       fmax = n392_sota$sota_clean$fmax,              wf1 = n392_sota$sota_clean$weighted_f1,              grp = "Baseline"),
  data.frame(name = "ECRECer",                     fmax = n392_sota$sota_ecrecer$fmax,            wf1 = n392_sota$sota_ecrecer$weighted_f1,            grp = "Baseline"),
  data.frame(name = "EnzymeHunter",                fmax = n392_sota$sota_enzymehunter$fmax,       wf1 = n392_sota$sota_enzymehunter$weighted_f1,       grp = "Baseline"),
  data.frame(name = "DeepECtransformer",           fmax = n392_sota$sota_deepec_transformer$fmax, wf1 = n392_sota$sota_deepec_transformer$weighted_f1, grp = "Baseline"),
  data.frame(name = "ZymeBind (Sequence only)",    fmax = n392_zb$ens4_seq_only$fmax,             wf1 = n392_zb$ens4_seq_only$weighted_f1,             grp = "Biophysics only"),
  data.frame(name = "ZymeBind (Seq + Structure)",  fmax = n392_zb$ens4_observed3di$fmax,          wf1 = n392_zb$ens4_observed3di$weighted_f1,          grp = "Biophysics only"),
  data.frame(name = "ZymeBind (<=2022 annotation)",fmax = n392_zb$ens4_frozen_2022_04$fmax,       wf1 = n392_zb$ens4_frozen_2022_04$weighted_f1,       grp = "ZymeBind"),
  data.frame(name = "ZymeBind",                    fmax = n392_zb$ens4_full_context$fmax,         wf1 = n392_zb$ens4_full_context$weighted_f1,         grp = "ZymeBind")
)
df_a$name <- factor(df_a$name, levels = df_a$name)

p_a <- ggplot(df_a, aes(x = fmax, y = name, fill = grp)) +
  geom_col(width = 0.66, alpha = 0.94) +
  geom_point(aes(x = wf1), color = "#1E293B", size = 1.65, shape = 18) +
  scale_fill_manual(values = c("Baseline" = "#9AA7B4", "Biophysics only" = COL_TEAL, "ZymeBind" = COL_HERO)) +
  scale_x_continuous(limits = c(0, 0.55), expand = c(0, 0), breaks = seq(0, 0.5, 0.1)) +
  annotate("text", x = 0.492, y = 10, label = "0.486", size = 2.0, family = "Arial", hjust = 0, fontface = "bold", color = COL_HERO) +
  labs(x = "NEW-392 Fmax (bar) and Weighted-F1 (diamond)", y = NULL, fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"), legend.text = element_text(size = 5.4, family = "Arial"))

# ---- Panel b: Price-149 Experimental Benchmark (consistent 4-model ensemble, zero Single/Ens version mixing) ----
p149_sota <- ext$benchmarks$price149$sota_same_split
p149_sota$sota_ecrecer <- ecr$external_benchmarks$price149
p149_zb <- ext$benchmarks$price149$zymebind_locked

df_b <- rbind(
  data.frame(name = "GraphEC-LD*",               Fmax = p149_sota$sota_graphec_ld$fmax,         wF1 = p149_sota$sota_graphec_ld$weighted_f1),
  data.frame(name = "HIT-EC",                    Fmax = p149_sota$sota_hitec$fmax,              wF1 = p149_sota$sota_hitec$weighted_f1),
  data.frame(name = "EnzymeHunter*",             Fmax = p149_sota$sota_enzymehunter$fmax,       wF1 = p149_sota$sota_enzymehunter$weighted_f1),
  data.frame(name = "CLEAN",                     Fmax = p149_sota$sota_clean$fmax,              wF1 = p149_sota$sota_clean$weighted_f1),
  data.frame(name = "DeepECtransformer",         Fmax = p149_sota$sota_deepec_transformer$fmax, wF1 = p149_sota$sota_deepec_transformer$weighted_f1),
  data.frame(name = "ECRECer",                   Fmax = p149_sota$sota_ecrecer$fmax,            wF1 = p149_sota$sota_ecrecer$weighted_f1),
  data.frame(name = "ZymeBind (Sequence only)",  Fmax = p149_zb$ens4_seq_only$fmax,             wF1 = p149_zb$ens4_seq_only$weighted_f1)
)
df_b$name <- factor(df_b$name, levels = df_b$name)
df_b_long <- pivot_longer(df_b, cols = c(Fmax, wF1), names_to = "metric", values_to = "val")

p_b <- ggplot(df_b_long, aes(x = val, y = name, fill = metric)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.66) +
  scale_fill_manual(values = c("Fmax" = COL_HERO, "wF1" = COL_TERRA), labels = c("Fmax", "Weighted-F1 (locked threshold)")) +
  scale_x_continuous(limits = c(0, 0.45), expand = c(0, 0), breaks = seq(0, 0.40, 0.1)) +
  annotate("text", x = 0.378, y = 7.18, label = "0.372", size = 2.0, family = "Arial", hjust = 0, fontface = "bold", color = COL_HERO) +
  labs(x = "Price-149 score (*requires 3D structure)", y = NULL, fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"), legend.text = element_text(size = 5.4, family = "Arial"))

# ---- Panel c: Stratification Across All 7 EC1 Classes ----
ec1 <- val_bs$stratification$ec1
ec1_names <- paste0("EC", 1:7)
df_c <- do.call(rbind, lapply(ec1_names, function(k) {
  d <- ec1[[k]]
  data.frame(
    ec1 = sprintf("%s\n(n=%d)", k, d$n),
    MMseqs2 = d$mmseqs2,
    `ZymeBind (Sequence only)` = d$zb_ens4_seq_only,
    ZymeBind = d$zb_ens4_full,
    check.names = FALSE, stringsAsFactors = FALSE
  )
}))
df_c$ec1 <- factor(df_c$ec1, levels = df_c$ec1)
df_c_long <- pivot_longer(df_c, cols = c(MMseqs2, `ZymeBind (Sequence only)`, ZymeBind), names_to = "method", values_to = "fmax")
df_c_long$method <- factor(df_c_long$method, levels = c("MMseqs2", "ZymeBind (Sequence only)", "ZymeBind"))

p_c <- ggplot(df_c_long, aes(x = ec1, y = fmax, fill = method)) +
  geom_col(position = position_dodge(width = 0.76), width = 0.70) +
  scale_fill_manual(values = c("MMseqs2" = COL_SLATE, "ZymeBind (Sequence only)" = COL_SKY, "ZymeBind" = COL_HERO)) +
  coord_cartesian(ylim = c(0.72, 1.02)) +
  scale_y_continuous(breaks = seq(0.75, 1.00, 0.05)) +
  labs(x = "Primary EC1 enzyme class", y = "Fmax", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"),
        legend.text = element_text(size = 5.3, family = "Arial"), axis.text.x = element_text(size = 5.6, family = "Arial"))

# ---- Panel d: Probability Calibration Reliability Diagram (consistent 4-model ensemble!) ----
bins_ens_full <- val_bs$calibration$zb_ens4_full$reliability_bins
bins_ens_seq  <- val_bs$calibration$zb_ens4_seq_only$reliability_bins

df_d <- rbind(
  data.frame(conf = bins_ens_full$conf, acc = bins_ens_full$acc, n = bins_ens_full$n,
             model = sprintf("ZymeBind (ECE = %.4f)", val_bs$calibration$zb_ens4_full$ece_temp_scaled_5fold)),
  data.frame(conf = bins_ens_seq$conf,  acc = bins_ens_seq$acc,  n = bins_ens_seq$n,
             model = sprintf("ZymeBind (Sequence only, ECE = %.4f)", val_bs$calibration$zb_ens4_seq_only$ece_temp_scaled_5fold))
) %>% filter(n >= 5)

p_d <- ggplot(df_d, aes(x = conf, y = acc, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.4) +
  geom_line(linewidth = 0.75) +
  geom_point(aes(size = log10(n + 1)), alpha = 0.9) +
  scale_color_manual(values = c(COL_HERO, COL_TEAL)) +
  scale_size_continuous(range = c(1.0, 2.6), guide = "none") +
  scale_x_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.2)) +
  labs(x = "Predicted confidence", y = "Observed top-1 accuracy", color = NULL) +
  theme(legend.position = "top", legend.direction = "vertical",
        legend.text = element_text(size = 5.4, family = "Arial"), legend.key.height = unit(7, "pt"))

fig3 <- wrap_plots(p_a, p_b, free(p_c, type = "label"), free(p_d, type = "label"), ncol = 2) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(fig3, file.path(FIG_DIR, "Fig3_external_strat_calib"), width_mm = 183, height_mm = 138, panel_ids = c("a", "b", "c", "d"))
