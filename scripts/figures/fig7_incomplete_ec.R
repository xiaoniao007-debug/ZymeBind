#!/usr/bin/env Rscript
# Figure 7: Large-Scale Completion of Incomplete EC Annotations (x.x.x.-, n = 3,245)
# Strict Arial font, unified nomenclature, no subpanel titles.

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
FIG_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures"

cs2 <- fromJSON(file.path(DATA_DIR, "case_study_2_partial_ec_and_audit.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_SKY   <- "#6B9AC4"

extract_hc <- function(mode_key, mode_label) {
  hc <- cs2$hierarchical_consistency[[mode_key]]
  thrs <- c("prob_ge_0.50", "prob_ge_0.70", "prob_ge_0.85")
  tvals <- c(">= 0.50", ">= 0.70", ">= 0.85")
  do.call(rbind, lapply(seq_along(thrs), function(i) {
    d <- hc[[thrs[i]]]
    data.frame(
      mode = mode_label, thr = tvals[i], n_comp = d$n_completed,
      ec3 = d$ec3_exact_consistency * 100, ec2 = d$ec2_consistency * 100,
      stringsAsFactors = FALSE
    )
  }))
}
df_hc <- rbind(extract_hc("seq_struct", "ZymeBind (Seq + Structure)"), extract_hc("all_4", "ZymeBind"))
df_hc$thr  <- factor(df_hc$thr,  levels = c(">= 0.50", ">= 0.70", ">= 0.85"))
df_hc$mode <- factor(df_hc$mode, levels = c("ZymeBind (Seq + Structure)", "ZymeBind"))

# ---- Panel a: EC3 Exact Consistency vs Confidence Gate ----
p_a <- ggplot(df_hc, aes(x = thr, y = ec3, color = mode, group = mode)) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.3) +
  geom_text(aes(label = sprintf("%.1f%%", ec3)), vjust = -0.9, size = 2.0, family = "Arial", fontface = "bold", show.legend = FALSE) +
  scale_color_manual(values = c("ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind" = COL_HERO)) +
  scale_y_continuous(limits = c(50, 100), breaks = seq(50, 100, 10)) +
  labs(x = "Calibrated probability threshold", y = "Exact EC3 consistency (%)", color = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.5, "pt"))

# ---- Panel b: EC2 vs EC3 Consistency at High Confidence (>= 0.85) ----
df_b <- df_hc %>%
  filter(thr == ">= 0.85") %>%
  pivot_longer(cols = c(ec3, ec2), names_to = "level", values_to = "pct") %>%
  mutate(level = factor(level, levels = c("ec3", "ec2"), labels = c("Exact EC3 match", "EC2 subclass match")))

p_b <- ggplot(df_b, aes(x = mode, y = pct, fill = level)) +
  geom_col(position = position_dodge(width = 0.65), width = 0.58) +
  geom_text(aes(label = sprintf("%.1f%%", pct)), position = position_dodge(width = 0.65),
            vjust = -0.4, size = 2.0, family = "Arial", fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = c("Exact EC3 match" = COL_HERO, "EC2 subclass match" = COL_SKY)) +
  scale_y_continuous(limits = c(0, 108), expand = c(0, 0), breaks = seq(0, 100, 20)) +
  labs(x = NULL, y = "Hierarchical consistency at tau >= 0.85 (%)", fill = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.5, "pt"))

# ---- Panel c: Precision-Coverage Trade-off ----
p_c <- ggplot(df_hc, aes(x = n_comp, y = ec3, color = mode, group = mode)) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.3) +
  geom_text_repel(aes(label = sprintf("%s\n(n = %d)", thr, n_comp)), size = 1.9, family = "Arial", box.padding = 0.35, show.legend = FALSE, seed = 42) +
  scale_color_manual(values = c("ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind" = COL_HERO)) +
  scale_x_continuous(limits = c(50, 740), breaks = seq(100, 700, 150)) +
  scale_y_continuous(limits = c(52, 98), breaks = seq(55, 95, 10)) +
  labs(x = "Completed incomplete-EC enzymes (n)", y = "Exact EC3 consistency (%)", color = NULL) +
  theme(legend.position = c(0.68, 0.86), legend.key.height = unit(7, "pt"))

# ---- Panel d: Top Held-Out Completions ----
top_c <- head(cs2$top_20_partial_completions, 10)
df_d <- data.frame(
  label = sprintf("%s: %s -> %s", top_c$accession, sapply(top_c$ec_partial, `[`, 1), top_c$pred_ec4_ss),
  prob_ss = top_c$prob_ss,
  prob_fl = top_c$prob_full,
  stringsAsFactors = FALSE
)
df_d$label <- factor(df_d$label, levels = rev(df_d$label))

p_d <- ggplot(df_d, aes(y = label)) +
  geom_segment(aes(x = pmin(prob_ss, prob_fl), xend = pmax(prob_ss, prob_fl), yend = label), color = "#CBD5E1", linewidth = 0.85) +
  geom_point(aes(x = prob_ss, color = "ZymeBind (Seq + Structure)"), size = 2.2) +
  geom_point(aes(x = prob_fl, color = "ZymeBind"), size = 2.2, shape = 17) +
  scale_color_manual(values = c("ZymeBind (Seq + Structure)" = COL_TEAL, "ZymeBind" = COL_HERO)) +
  scale_x_continuous(limits = c(0.985, 1.002), breaks = seq(0.985, 1.000, 0.005)) +
  labs(x = "Predicted EC4 probability", y = NULL, color = NULL) +
  theme(legend.position = "top", legend.key.size = unit(6.5, "pt"), axis.text.y = element_text(size = 5.4, family = "Arial"))

fig7 <- wrap_plots(p_a, free(p_b, type = "label"), p_c, p_d, ncol = 2) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

save_nature_fig(fig7, file.path(FIG_DIR, "Fig7_incomplete_ec_completion"), width_mm = 183, height_mm = 134, panel_ids = c("a", "b", "c", "d"))
