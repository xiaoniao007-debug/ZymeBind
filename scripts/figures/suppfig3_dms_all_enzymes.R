#!/usr/bin/env Rscript
# Supplementary Figure 3: Remaining 3 ProteinGym DMS Enzymes Mirrored Tracks + 4-Enzyme Mutant Correlation Scatter
# Strict Arial font, unified calm palette, no subpanel titles (a-d only).

source("/Users/chenchen/Paper/ZymeBind/scripts/pub_style.R")

DATA_DIR <- "/Users/chenchen/Paper/ZymeBind/data"
OUT_DIR  <- "/Users/chenchen/Paper/ZymeBind/figures/supplementary"

dms <- fromJSON(file.path(DATA_DIR, "ext2c_token_vs_pooled_dms.json"), simplifyVector = TRUE)

COL_HERO  <- "#2B5C8F"
COL_TEAL  <- "#4D908E"
COL_TERRA <- "#D97A54"

make_mirror_track <- function(bm_id, tag_lbl) {
  row <- dms$benchmarks[dms$benchmarks$id == bm_id, ]
  pr  <- row$per_residue[[1]]
  dms_inv <- max(pr$mean_dms_score, na.rm = TRUE) - pr$mean_dms_score
  df  <- data.frame(pos = pr$pos, dms_lof = dms_inv, plm_sens = -pr$mean_mm_score, is_act = pr$is_active)
  df$dms_norm <- df$dms_lof / max(df$dms_lof, na.rm = TRUE)
  df$plm_norm <- pmax(0, df$plm_sens) / max(pmax(0, df$plm_sens), na.rm = TRUE)
  act_pos <- df$pos[df$is_act]
  p <- ggplot(df, aes(x = pos))
  if (length(act_pos) > 0) {
    df_act <- data.frame(pos = act_pos)
    p <- p + geom_rect(data = df_act, aes(xmin = pos - 2.5, xmax = pos + 2.5, ymin = -1.06, ymax = 1.06),
                       inherit.aes = FALSE, fill = "#00C4CC", alpha = 0.22)
  }
  p +
    geom_area(aes(y =  dms_norm), fill = COL_HERO, alpha = 0.60) +
    geom_line(aes(y =  dms_norm), color = "#1E3A5F", linewidth = 0.30) +
    geom_area(aes(y = -plm_norm), fill = COL_TERRA, alpha = 0.60) +
    geom_line(aes(y = -plm_norm), color = "#9A482A", linewidth = 0.30) +
    geom_hline(yintercept = 0, color = "#1E293B", linewidth = 0.40) +
    annotate("text", x = min(df$pos) + 3, y = 1.15, label = tag_lbl,
             hjust = 0, size = 1.95, family = "Arial", fontface = "bold", color = "#1E3A5F") +
    scale_y_continuous(limits = c(-1.08, 1.25), breaks = c(-1, -0.5, 0, 0.5, 1), labels = c("1.0", "0.5", "0", "0.5", "1.0")) +
    labs(x = "Residue position (amino acids)", y = "Normalized sensitivity")
}

p_a <- make_mirror_track("KKA2_KLEPN_Melnikov_2014", "APH(3')-II (EC 2.7.1.95, per-residue rho = 0.757)")
p_b <- make_mirror_track("MTH3_HAEAE_RockahShmuel_2015", "M.HaeIII (EC 2.1.1.37, per-residue rho = 0.579)")
p_c <- make_mirror_track("OXDA_RHOTO_Vanella_2023_activity", "DAOx (EC 1.4.3.3, per-residue rho = 0.501)")

# ---- Panel d: 4-Enzyme Mutant-Level Scatter (Experimental DMS vs ZymeBind Score) ----
bm <- dms$benchmarks
short_nm <- c("TEM-1 beta-lactamase" = "TEM-1 (rho = 0.732)", "DNA methyltransferase M.HaeIII" = "M.HaeIII (rho = 0.527)",
              "Aminoglycoside kinase APH(3')-II" = "APH(3')-II (rho = 0.601)", "D-amino acid oxidase (DAOx)" = "DAOx (rho = 0.370)")
sc_list <- list()
for (i in seq_len(nrow(bm))) {
  ss <- bm$scatter_sample[[i]]
  sc_list[[i]] <- data.frame(
    enz = short_nm[[bm$name[i]]],
    mm  = ss$mm_score,
    dms = ss$dms_score,
    stringsAsFactors = FALSE
  )
}
df_d <- do.call(rbind, sc_list)
df_d$enz <- factor(df_d$enz, levels = c("TEM-1 (rho = 0.732)", "APH(3')-II (rho = 0.601)", "M.HaeIII (rho = 0.527)", "DAOx (rho = 0.370)"))

p_d <- ggplot(df_d, aes(x = mm, y = dms, color = enz)) +
  geom_point(size = 0.55, alpha = 0.42) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.65, color = "#1E293B") +
  facet_wrap(~ enz, scales = "free", ncol = 4) +
  scale_color_manual(values = c(COL_HERO, COL_TEAL, "#6B9AC4", COL_TERRA), guide = "none") +
  labs(x = "ZymeBind PLM masked-marginal log-odds", y = "Experimental DMS fitness") +
  theme(strip.text = element_text(size = 5.6, family = "Arial", face = "bold"))

sfig3 <- (p_a / p_b / p_c / p_d) +
  plot_layout(heights = c(0.85, 0.85, 0.85, 1.05)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(size = 8.5, face = "bold", family = "Arial"))

w <- 183 / 25.4; h <- 206 / 25.4
out_p <- file.path(OUT_DIR, "SuppFig3_dms_all_enzymes_tracks_scatter")
svglite::svglite(paste0(out_p, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial")); print(sfig3); grDevices::dev.off()
grDevices::quartz(type = "pdf", file = paste0(out_p, ".pdf"), width = w, height = h, family = "Arial"); print(sfig3); grDevices::dev.off()
ragg::agg_png(paste0(out_p, ".png"), width = w, height = h, units = "in", res = 300, background = "white"); print(sfig3); grDevices::dev.off()
ragg::agg_tiff(paste0(out_p, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(sfig3); grDevices::dev.off()
cat("[EXPORTED] SuppFig3 (183 x 206 mm, Arial)\n")
