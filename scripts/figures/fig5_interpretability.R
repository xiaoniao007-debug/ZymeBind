#!/usr/bin/env Rscript
# Figure 5: Multi-Level Mechanistic Interpretability (Catalytic Masking, Shapley Flow & Sparse Autoencoder)
# Strict Arial font, fixed Panel b y-axis title distance, full scientific labels (zero EC3 Mech. / ACT_SITE), no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
FIG_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures"

l1  <- fromJSON(file.path(DATA_DIR, "interp_l1_site_masking_v2_p1m1_test.json"), simplifyVector = TRUE)
mod <- fromJSON(file.path(DATA_DIR, "val_modality_matrix_and_shapley.json"), simplifyVector = TRUE)
l3  <- fromJSON(file.path(DATA_DIR, "val_interpretability_l3_sae_monosemantic.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"
COL_ICE   <- "#C6D9EC"

# ---- Panel a: Per-Protein Catalytic & Binding Site Masking (n = 400) ----
pa <- l1$per_protein$act_site
pb <- l1$per_protein$act_or_bind
df_pts <- rbind(
  data.frame(cond = "Catalytic site\n(n = 400)",               grp = "Functional site", drop = pa$drop_site * 1000),
  data.frame(cond = "Catalytic site\n(n = 400)",               grp = "Random control",  drop = pa$drop_rand_mean * 1000),
  data.frame(cond = "Catalytic + binding\n(n = 400)",          grp = "Functional site", drop = pb$drop_site * 1000),
  data.frame(cond = "Catalytic + binding\n(n = 400)",          grp = "Random control",  drop = pb$drop_rand_mean * 1000),
  data.frame(cond = "Catalytic + binding\n(correct, n = 244)", grp = "Functional site", drop = pb$drop_site[pb$correct] * 1000),
  data.frame(cond = "Catalytic + binding\n(correct, n = 244)", grp = "Random control",  drop = pb$drop_rand_mean[pb$correct] * 1000)
)
df_pts$cond <- factor(df_pts$cond, levels = c(
  "Catalytic site\n(n = 400)", "Catalytic + binding\n(n = 400)", "Catalytic + binding\n(correct, n = 244)"
))
df_pts$grp <- factor(df_pts$grp, levels = c("Functional site", "Random control"))
df_pts$drop_vis <- pmin(pmax(df_pts$drop, -3), 34)

s1 <- l1$results$act_site$all
s2 <- l1$results$act_or_bind$all
s3 <- l1$results$act_or_bind$pred_correct
se_fn <- function(v) sd(v) / sqrt(length(v))
df_sum <- data.frame(
  cond = rep(levels(df_pts$cond), each = 2),
  grp  = factor(rep(c("Functional site", "Random control"), 3), levels = c("Functional site", "Random control")),
  mean = c(s1$mean_drop_site, s1$mean_drop_random, s2$mean_drop_site, s2$mean_drop_random, s3$mean_drop_site, s3$mean_drop_random) * 1000,
  se   = c(se_fn(pa$drop_site), se_fn(pa$drop_rand_mean), se_fn(pb$drop_site), se_fn(pb$drop_rand_mean), se_fn(pb$drop_site[pb$correct]), se_fn(pb$drop_rand_mean[pb$correct])) * 1000
)
df_sum$cond <- factor(df_sum$cond, levels = levels(df_pts$cond))

df_ann <- data.frame(
  cond = levels(df_pts$cond),
  y    = c(17.5, 27.5, 29.5),
  lbl  = c(sprintf("%.1fx (p = %.1e)", s1$mean_drop_site / s1$mean_drop_random, s1$wilcoxon_greater_p),
           sprintf("%.1fx (p = %.1e)", s2$mean_drop_site / s2$mean_drop_random, s2$wilcoxon_greater_p),
           sprintf("%.1fx (p = %.1e)", s3$mean_drop_site / s3$mean_drop_random, s3$wilcoxon_greater_p))
)

p_a <- ggplot() +
  geom_col(data = df_sum, aes(x = cond, y = mean, fill = grp),
           position = position_dodge(width = 0.72), width = 0.64, alpha = 0.88) +
  geom_point(data = df_pts, aes(x = cond, y = drop_vis, color = grp),
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.72, seed = 7),
             size = 0.50, alpha = 0.28, show.legend = FALSE) +
  geom_errorbar(data = df_sum, aes(x = cond, ymin = mean - se, ymax = mean + se, group = grp),
                position = position_dodge(width = 0.72), width = 0.20, linewidth = 0.50, color = "#1E293B") +
  geom_label(data = df_ann, aes(x = cond, y = y, label = lbl),
             size = 1.75, family = "Arial", fontface = "bold", color = "#1E3A5F",
             fill = alpha("#FFFFFF", 0.85), label.size = 0, label.padding = unit(1.2, "pt")) +
  scale_fill_manual(values = c("Functional site" = COL_HERO, "Random control" = "#CBD5E1")) +
  scale_color_manual(values = c("Functional site" = "#1E3A5F", "Random control" = "#64748B")) +
  coord_cartesian(ylim = c(-2.5, 34)) +
  labs(x = NULL, y = "True-EC4 probability drop (\u00D7 10\u207B\u00B3)", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"),
        legend.text = element_text(size = 5.4, family = "Arial"), axis.text.x = element_text(size = 5.3, family = "Arial"))

# ---- Panel b: Exact 2^4 Shapley Attribution Flow (free y-label so title sits snug to y-axis!) ----
sh_df <- data.frame(
  stage = rep(1:5, each = 4),
  mod   = rep(c("Sequence", "Structure", "Text", "Domain"), 5),
  pct   = c(
    11, 9, 15, 65,
    35, 14, 13, 38,
    30, 21, 22, 27,
    36, 8, 29, 28,
    88, 0, 0, 12
  ),
  stringsAsFactors = FALSE
)
sh_df$mod <- factor(sh_df$mod, levels = c("Domain", "Text", "Structure", "Sequence"))
sh_df <- sh_df %>%
  arrange(stage, desc(mod)) %>%
  group_by(stage) %>%
  mutate(y1 = cumsum(pct), y0 = y1 - pct) %>%
  ungroup()

make_rib <- function(x0, x1, a0, a1, b0, b1, mod_name, pid, n = 28) {
  t <- seq(0, 1, length.out = n); s <- 3 * t^2 - 2 * t^3
  xt <- x0 + (x1 - x0) * t
  data.frame(x = c(xt, rev(xt)), y = c(a1 + (b1 - a1) * s, rev(a0 + (b0 - a0) * s)), mod = mod_name, pair_id = pid)
}
alluv_list <- list()
for (st in 1:4) {
  for (m_nm in levels(sh_df$mod)) {
    r0 <- sh_df %>% filter(stage == st, mod == m_nm)
    r1 <- sh_df %>% filter(stage == st + 1, mod == m_nm)
    alluv_list[[length(alluv_list) + 1]] <- make_rib(st + 0.24, st + 0.76, r0$y0, r0$y1, r1$y0, r1$y1, m_nm, paste0(m_nm, "_", st))
  }
}
df_alluv <- do.call(rbind, alluv_list)
df_alluv$mod <- factor(df_alluv$mod, levels = levels(sh_df$mod))

mod_cols <- c("Sequence" = COL_HERO, "Structure" = COL_TEAL, "Text" = COL_TERRA, "Domain" = "#A9B4C2")
stage_lbls <- c("Standard\nfusion", "+ Modality\ndropout", "ZymeBind\n(All EC4)", "Medium EC4\n(6-20 train)", "Rare EC4\n(1-5 train)")

p_b <- ggplot() +
  geom_polygon(data = df_alluv, aes(x = x, y = y, fill = mod, group = pair_id), alpha = 0.26) +
  geom_rect(data = sh_df, aes(xmin = stage - 0.24, xmax = stage + 0.24, ymin = y0, ymax = y1, fill = mod),
            color = "#FFFFFF", linewidth = 0.45) +
  geom_text(data = sh_df %>% filter(pct >= 11),
            aes(x = stage, y = 0.5 * (y0 + y1), label = paste0(pct, "%")),
            size = 1.72, family = "Arial", fontface = "bold", color = "#FFFFFF") +
  scale_fill_manual(values = mod_cols) +
  scale_x_continuous(breaks = 1:5, labels = stage_lbls, expand = expansion(mult = c(0.05, 0.05))) +
  scale_y_continuous(limits = c(0, 102), expand = c(0, 0), breaks = seq(0, 100, 25)) +
  labs(x = NULL, y = "Exact Shapley attribution (%)", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"),
        legend.text = element_text(size = 5.4, family = "Arial"),
        axis.text.x = element_text(size = 5.2, family = "Arial"))

# ---- Panel c: Top-K Sparse Autoencoder Concept Alignment (zero casual abbreviations!) ----
df_c <- data.frame(
  grp = factor(c("InterPro domains\n(n = 50)", "EC3 sub-subclasses\n(n = 48)", "Catalytic cofactors\n(n = 10)"),
               levels = c("InterPro domains\n(n = 50)", "EC3 sub-subclasses\n(n = 48)", "Catalytic cofactors\n(n = 10)")),
  AUROC = c(l3$interpro_domains$mean_auroc, l3$ec3_mechanisms$mean_auroc, l3$cofactors$mean_auroc),
  F1    = c(l3$interpro_domains$mean_f1,    l3$ec3_mechanisms$mean_f1,    l3$cofactors$mean_f1),
  Frac  = c(l3$interpro_domains$frac_f1_ge_0.7, l3$ec3_mechanisms$frac_f1_ge_0.7, l3$cofactors$frac_f1_ge_0.7)
) %>% pivot_longer(cols = c(AUROC, F1, Frac), names_to = "m", values_to = "v")
df_c$m <- factor(df_c$m, levels = c("AUROC", "F1", "Frac"), labels = c("Mean AUROC", "Mean F1", "Fraction F1 >= 0.7"))

p_c <- ggplot(df_c, aes(x = grp, y = v, fill = m)) +
  geom_col(position = position_dodge(width = 0.74), width = 0.66) +
  geom_text(aes(label = sprintf("%.2f", v)), position = position_dodge(width = 0.74),
            vjust = -0.35, size = 1.80, family = "Arial", color = "#1E293B") +
  scale_fill_manual(values = c("Mean AUROC" = COL_HERO, "Mean F1" = COL_TEAL, "Fraction F1 >= 0.7" = COL_TERRA)) +
  scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0), breaks = seq(0, 1.0, 0.2)) +
  labs(x = NULL, y = "Sparse autoencoder alignment score", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.2, "pt"),
        legend.text = element_text(size = 5.2, family = "Arial"),
        axis.text.x = element_text(size = 5.5, family = "Arial"))

# ---- Panel d: Top Monosemantic Single-Neuron Alignment Dumbbells ----
get_top4 <- function(tbl, cat_lbl) {
  t4 <- head(tbl, 4)
  data.frame(
    name = sprintf("%s (#%d)", t4$concept, t4$feature_id),
    f1 = t4$f1, auroc = t4$auroc, cat = cat_lbl, stringsAsFactors = FALSE
  )
}
df_d <- rbind(
  get_top4(l3$interpro_domains$top10, "InterPro domain"),
  get_top4(l3$ec3_mechanisms$top10,   "EC3 sub-subclass"),
  get_top4(l3$cofactors$top10,        "Catalytic cofactor")
)
df_d$name <- factor(df_d$name, levels = rev(df_d$name))

p_d <- ggplot(df_d) +
  geom_segment(aes(x = 0.48, xend = auroc, y = name, yend = name), color = "#E2E8F0", linewidth = 0.5) +
  geom_segment(aes(x = f1, xend = auroc, y = name, yend = name, color = cat), linewidth = 1.15, alpha = 0.85) +
  geom_point(aes(x = f1, y = name, color = cat, shape = "F1 score"), size = 2.3) +
  geom_point(aes(x = auroc, y = name, color = cat, shape = "AUROC"), size = 2.5, fill = "#FFFFFF", stroke = 0.85) +
  scale_color_manual(values = c("InterPro domain" = COL_HERO, "EC3 sub-subclass" = COL_TEAL, "Catalytic cofactor" = COL_TERRA)) +
  scale_shape_manual(values = c("F1 score" = 16, "AUROC" = 21)) +
  guides(color = guide_legend(override.aes = list(size = 2.0)),
         shape = guide_legend(override.aes = list(size = 2.0))) +
  scale_x_continuous(limits = c(0.48, 1.03), breaks = seq(0.5, 1.0, 0.1)) +
  labs(x = "Single-neuron concept alignment", y = NULL, color = NULL, shape = NULL) +
  theme(legend.position = c(0.28, 0.30), legend.box = "vertical", legend.spacing.y = unit(2, "pt"),
        legend.key.height = unit(6.5, "pt"), legend.text = element_text(size = 5.1, family = "Arial"),
        axis.text.y = element_text(size = 5.4, family = "Arial"))

# Free both p_b and p_c y-label alignment so p_b's y-axis title sits snug against its axis!
fig5 <- wrap_plots(p_a, free(p_b, type = "label"), free(p_c, type = "label"), p_d, ncol = 2) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(fig5, file.path(FIG_DIR, "Fig5_interpretability_L1_L2_L3"), width_mm = 183, height_mm = 140, panel_ids = c("a", "b", "c", "d"))
