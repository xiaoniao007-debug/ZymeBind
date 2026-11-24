#!/usr/bin/env Rscript
# Figure 4: Component Ablations, 9-Modality Synergy Matrix & Zero-Shot EC Retrieval
# Strict Arial font, unified nomenclature, fixed Panel a top-bar label overlap, no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
FIG_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures"

abl <- fromJSON(file.path(DATA_DIR, "val_ablation_p1m1_8metrics.json"), simplifyVector = TRUE)
mod <- fromJSON(file.path(DATA_DIR, "val_modality_matrix_and_shapley.json"), simplifyVector = TRUE)
m1  <- fromJSON(file.path(DATA_DIR, "val_mode1_zeroshot_ablation.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"
COL_TERRA <- "#D97A54"
COL_SLATE <- "#8D99AE"

# ---- Panel a: Controlled Single-Variable Ablation (zero overlap at top bar!) ----
base_full <- abl$metrics$v4_full$full$fmax
base_ss   <- abl$metrics$v4_full$seq_struct$fmax
base_seq  <- abl$metrics$v4_full$seq_only$fmax

abl_map <- c(
  "no_moddrop" = "- Modality dropout",
  "focal_loss" = "Focal loss (vs ASL)",
  "no_prostt5" = "- ProstT5-3B encoder",
  "no_permod"  = "- Modality-specific heads",
  "no_kd"      = "- Cross-modal distillation"
)

df_a <- do.call(rbind, lapply(names(abl_map), function(k) {
  m <- abl$metrics[[k]]
  s <- abl$significance[[paste0("v4_full_minus_", k)]]
  rbind(
    data.frame(ablation = abl_map[[k]], view = "ZymeBind",                   delta = m$full$fmax - base_full,     pval = s$full$p_one_sided),
    data.frame(ablation = abl_map[[k]], view = "ZymeBind (Seq + Structure)", delta = m$seq_struct$fmax - base_ss, pval = s$seq_struct$p_one_sided),
    data.frame(ablation = abl_map[[k]], view = "ZymeBind (Sequence only)",   delta = m$seq_only$fmax - base_seq,  pval = s$seq_only$p_one_sided)
  )
}))
df_a$ablation <- factor(df_a$ablation, levels = rev(unname(abl_map)))
df_a$view <- factor(df_a$view, levels = c("ZymeBind", "ZymeBind (Seq + Structure)", "ZymeBind (Sequence only)"))
df_a$sig_star <- ifelse(df_a$pval <= 0.01, "**", ifelse(df_a$pval <= 0.05, "*", ""))

p_a <- ggplot(df_a, aes(x = delta, y = ablation, fill = view)) +
  geom_vline(xintercept = 0, color = "#334155", linewidth = 0.40, linetype = "dashed") +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  geom_text(aes(x = delta - 0.007, label = sig_star),
            position = position_dodge(width = 0.72), hjust = 1, size = 2.3, family = "Arial", color = "#111111") +
  scale_fill_manual(values = c("ZymeBind" = COL_HERO, "ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind (Sequence only)" = COL_SKY)) +
  scale_x_continuous(limits = c(-0.39, 0.025), breaks = seq(-0.35, 0.0, 0.07)) +
  labs(x = "Fmax change upon component removal (*p < 0.05, **p < 0.01)", y = NULL, fill = NULL) +
  theme(legend.position = c(0.36, 0.25), legend.key.height = unit(7, "pt"),
        legend.text = element_text(size = 5.3, family = "Arial"))

# ---- Panel b: 9-Combination Modality Synergy Matrix ----
mod_map <- c(
  "struct_only"       = "Structure only",
  "text_only"         = "Text only",
  "domain_only"       = "Domain only",
  "seq_only"          = "Sequence only",
  "seq_struct"        = "Seq + Structure",
  "seq_domain"        = "Seq + Domain",
  "seq_struct_domain" = "Seq + Structure + Domain",
  "seq_text"          = "Seq + Text",
  "all_4_modalities"  = "ZymeBind (All modalities)"
)

df_b <- do.call(rbind, lapply(names(mod_map), function(k) {
  d <- mod$modality_matrix[[k]]
  data.frame(
    combo = mod_map[[k]], Fmax = d$fmax, `AUPR-macro` = d$aupr_macro,
    `Macro-F1` = d$f1_macro_L4, `Rare EC4` = d$by_train_freq$rare_1_5$fmax,
    check.names = FALSE, stringsAsFactors = FALSE
  )
}))
df_b$combo <- factor(df_b$combo, levels = unname(mod_map))
df_b_long <- pivot_longer(df_b, cols = c(Fmax, `AUPR-macro`, `Macro-F1`, `Rare EC4`), names_to = "metric", values_to = "val")
df_b_long$metric <- factor(df_b_long$metric, levels = c("Fmax", "AUPR-macro", "Macro-F1", "Rare EC4"))

p_b <- ggplot(df_b_long, aes(x = metric, y = combo, fill = val)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.3f", val), color = val > 0.65), size = 1.9, family = "Arial", show.legend = FALSE) +
  scale_fill_gradientn(colors = c("#EEF5FB", "#C6D9EC", "#6B9AC4", "#2B5C8F", "#1E3A5F"), limits = c(0.05, 0.98)) +
  scale_color_manual(values = c("FALSE" = "#1E293B", "TRUE" = "white")) +
  labs(x = NULL, y = NULL, fill = "Score") +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "right", legend.key.width = unit(6, "pt"), legend.key.height = unit(28, "pt"))

# ---- Panel c: Zero-Shot Unseen-EC4 Retrieval Ablation ----
m1_map <- c(
  "no_biobert_sem" = "- BioBERT\nsemantics",
  "mode1_full"     = "Ontology\nGCN",
  "no_supcon"      = "- Contrastive\nloss",
  "no_gcn"         = "ZymeBind\n(no GCN)"
)

df_c <- do.call(rbind, lapply(names(m1_map), function(k) {
  v <- m1$ablations[[k]]$views
  rbind(
    data.frame(variant = m1_map[[k]], view = "ZymeBind",                   r10 = v$full[["recall@10"]]),
    data.frame(variant = m1_map[[k]], view = "ZymeBind (Seq + Structure)", r10 = v$seq_struct[["recall@10"]]),
    data.frame(variant = m1_map[[k]], view = "ZymeBind (Sequence only)",   r10 = v$seq_only[["recall@10"]])
  )
}))
df_c$variant <- factor(df_c$variant, levels = unname(m1_map))
df_c$view <- factor(df_c$view, levels = c("ZymeBind", "ZymeBind (Seq + Structure)", "ZymeBind (Sequence only)"))

p_c <- ggplot(df_c, aes(x = variant, y = r10, fill = view)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  scale_fill_manual(values = c("ZymeBind" = COL_HERO, "ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind (Sequence only)" = COL_SKY)) +
  scale_y_continuous(limits = c(0, 0.47), expand = c(0, 0), breaks = seq(0, 0.45, 0.1)) +
  annotate("text", x = 3.76, y = 0.435, label = "0.416", size = 2.0, family = "Arial", fontface = "bold", color = COL_HERO) +
  labs(x = NULL, y = "Zero-shot Recall@10 (376 unseen EC4s)", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6, "pt"), legend.text = element_text(size = 5.1, family = "Arial"),
        axis.text.x = element_text(size = 5.6, family = "Arial"))

# ---- Panel d: Effect of Removing Ontology GCN Over-Smoothing ----
df_d <- do.call(rbind, lapply(names(m1_map), function(k) {
  v <- m1$ablations[[k]]$views$full
  data.frame(
    variant = gsub("\n", " ", m1_map[[k]]),
    r100 = v[["recall@100"]],
    med_rank = v$median_rank,
    stringsAsFactors = FALSE
  )
}))
df_d$lx <- c(23.0, 20.6, 14.8, 14.5)
df_d$ly <- c(0.848, 0.888, 0.858, 0.940)

p_d <- ggplot(df_d, aes(x = med_rank, y = r100, color = variant)) +
  annotate("segment", x = 19.8, y = 0.838, xend = 13.4, yend = 0.960,
           arrow = arrow(length = unit(4.5, "pt")), color = COL_TERRA, linewidth = 0.65, linetype = "dashed") +
  geom_point(size = 3.1) +
  geom_text(aes(x = lx, y = ly, label = sprintf("%s\n(R@100 = %.3f, Median = %d)", variant, r100, as.integer(med_rank))),
            size = 1.85, family = "Arial", lineheight = 0.92, show.legend = FALSE) +
  scale_color_manual(values = c("- BioBERT semantics" = COL_SLATE, "Ontology GCN" = COL_HERO, "- Contrastive loss" = COL_TEAL, "ZymeBind (no GCN)" = COL_TERRA), guide = "none") +
  scale_x_reverse(limits = c(26.5, 10.5), breaks = seq(25, 11, -3)) +
  scale_y_continuous(limits = c(0.79, 0.99), breaks = seq(0.80, 0.98, 0.04)) +
  labs(x = "Median retrieval rank (lower is better)", y = "Zero-shot Recall@100")

fig4 <- wrap_plots(p_a, p_b, free(p_c, type = "label"), free(p_d, type = "label"), ncol = 2) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(fig4, file.path(FIG_DIR, "Fig4_ablations_zeroshot"), width_mm = 183, height_mm = 138, panel_ids = c("a", "b", "c", "d"))
