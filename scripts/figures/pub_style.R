# Shared Nature Portfolio R Publication Style & Alignment-Gated Export
# Unified Arial typography + calm editorial palette + no verbose panel titles (a, b, c, d only)

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(ggrepel)
})

source("/Users/chenchen/.workbuddy/skills/nature-figure/scripts/panel_alignment.R")

PAL_MODEL <- c(
  "ZymeBind"                   = "#2B5C8F",
  "ZymeBind (Seq + Structure)" = "#4D908E",
  "ZymeBind (Sequence only)"   = "#6B9AC4",
  "ZymeBind (<=2022 cutoff)"   = "#5B8DBE",
  "GraphEC-LD"                 = "#7B9EC2",
  "DeepECtransformer"          = "#D97A54",
  "ECRECer"                    = "#88A9A4",
  "EnzymeHunter"               = "#C89D8B",
  "HIT-EC"                     = "#B8A9C9",
  "CLEAN"                      = "#9EB3C2",
  "Dual-PLM kNN"               = "#BCC4CE",
  "MMseqs2"                    = "#8D99AE"
)

PAL_MODALITY <- c(
  "Sequence"  = "#2B5C8F",
  "Structure" = "#4D908E",
  "Text"      = "#D97A54",
  "Domain"    = "#A9B4C2"
)

PAL_DIRECTION <- c(
  "Gain"    = "#4D908E",
  "Drop"    = "#D97A54",
  "Neutral" = "#8D99AE"
)

theme_nature <- function(base_size = 7.2) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      text              = element_text(family = "Arial", colour = "#1E293B"),
      axis.line         = element_line(linewidth = 0.35, colour = "#222222"),
      axis.ticks        = element_line(linewidth = 0.35, colour = "#222222"),
      axis.ticks.length = unit(2.2, "pt"),
      axis.text         = element_text(size = 6.4, family = "Arial", colour = "#222222"),
      axis.title        = element_text(size = 7.2, family = "Arial", colour = "#111111", face = "plain"),
      plot.title        = element_blank(),
      plot.subtitle     = element_blank(),
      legend.title      = element_text(size = 6.5, family = "Arial", face = "bold", colour = "#111111"),
      legend.text       = element_text(size = 6.2, family = "Arial", colour = "#222222"),
      legend.key.size   = unit(7.5, "pt"),
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      strip.background  = element_blank(),
      strip.text        = element_text(size = 6.6, family = "Arial", face = "bold", colour = "#111111"),
      panel.grid        = element_blank(),
      plot.background   = element_rect(fill = "white", colour = NA),
      panel.background  = element_rect(fill = "white", colour = NA),
      plot.margin       = margin(4, 6, 4, 4, "pt"),
      plot.tag          = element_text(size = 8.5, family = "Arial", face = "bold", colour = "#111111")
    )
}

theme_set(theme_nature())

.resolve_gtable_pt <- function(units_vec, total_pt, axis = c("w", "h")) {
  axis <- match.arg(axis)
  n <- length(units_vec)
  vals <- numeric(n)
  null_w <- numeric(n)
  for (i in seq_len(n)) {
    u <- units_vec[i]
    txt <- as.character(u)
    if (grepl("null$", txt)) {
      null_w[i] <- as.numeric(sub("null$", "", txt))
    } else {
      vals[i] <- if (axis == "w") grid::convertWidth(u, "pt", valueOnly = TRUE) else grid::convertHeight(u, "pt", valueOnly = TRUE)
    }
  }
  rem <- max(total_pt - sum(vals), 0)
  if (sum(null_w) > 0) {
    vals[null_w > 0] <- rem * (null_w[null_w > 0] / sum(null_w))
  }
  vals
}

save_nature_fig <- function(plot, out_prefix, width_mm = 183, height_mm = 135,
                            dpi_png = 300, dpi_tiff = 600,
                            panel_ids = NULL, row_groups = NULL,
                            column_groups = NULL, exemptions = list()) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4

  if (inherits(plot, "patchwork")) {
    tmp_png <- tempfile(fileext = ".png")
    ragg::agg_png(tmp_png, width = w, height = h, units = "in", res = 150)
    grob <- patchwork::patchworkGrob(plot)
    grDevices::dev.off(); unlink(tmp_png)
    panel_rows <- grob$layout[grepl("^panel(?:-[0-9]+)?$", grob$layout$name, perl = TRUE), , drop = FALSE]
    panel_rows <- panel_rows[order(panel_rows$t, panel_rows$l), , drop = FALSE]
    if (is.null(panel_ids)) panel_ids <- letters[seq_len(nrow(panel_rows))]

    probe <- tempfile(fileext = ".pdf")
    grDevices::quartz(type = "pdf", file = probe, width = w, height = h, family = "Arial")
    grid::grid.newpage(); grid::grid.draw(grob); grid::grid.force()
    w_pt <- .resolve_gtable_pt(grob$widths,  w * 72, "w")
    h_pt <- .resolve_gtable_pt(grob$heights, h * 72, "h")
    grDevices::dev.off(); unlink(probe)

    x_edges <- c(0, cumsum(w_pt))
    y_edges <- c(0, cumsum(h_pt))
    tot_h   <- sum(h_pt)
    panels <- lapply(seq_len(nrow(panel_rows)), function(idx) {
      rw <- panel_rows[idx, ]
      list(
        id = panel_ids[idx],
        bbox_pt = unname(c(x_edges[rw$l], tot_h - y_edges[rw$b + 1], x_edges[rw$r + 1], tot_h - y_edges[rw$t])),
        grid_id = "patchwork-grid-1",
        row_start = as.integer(rw$t - 1), row_stop = as.integer(rw$b),
        col_start = as.integer(rw$l - 1), col_stop = as.integer(rw$r)
      )
    })
    manifest_path <- paste0(out_prefix, ".alignment-layout.json")
    report_path   <- paste0(out_prefix, ".alignment.json")
    overlay_svg   <- paste0(out_prefix, ".alignment.svg")
    jsonlite::write_json(
      list(schema_version = 1L, backend = "r-patchwork",
           figure = list(width_pt = w * 72, height_pt = h * 72),
           panels = panels, exemptions = exemptions),
      manifest_path, auto_unbox = TRUE, pretty = TRUE
    )
    status <- system2(
      "python3",
      c("/Users/chenchen/.workbuddy/skills/nature-figure/scripts/audit_panel_alignment.py",
        manifest_path, "--tolerance-pt", "1.5", "--gutter-tolerance-pt", "1.5",
        "--json-out", report_path, "--overlay-svg", overlay_svg)
    )
    if (status != 0L) stop(sprintf("Panel alignment gate FAILED (exit %d): %s", status, report_path), call. = FALSE)
  }

  svglite::svglite(paste0(out_prefix, ".svg"), width = w, height = h, system_fonts = list(sans = "Arial"))
  print(plot)
  grDevices::dev.off()

  grDevices::quartz(type = "pdf", file = paste0(out_prefix, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  grDevices::dev.off()

  ragg::agg_png(paste0(out_prefix, ".png"), width = w, height = h, units = "in", res = dpi_png, background = "white")
  print(plot)
  grDevices::dev.off()

  ragg::agg_tiff(paste0(out_prefix, ".tiff"), width = w, height = h, units = "in", res = dpi_tiff, compression = "lzw", background = "white")
  print(plot)
  grDevices::dev.off()

  cat(sprintf("[EXPORTED] %s.{pdf,svg,png,tiff} (%.0f x %.0f mm)\n", out_prefix, width_mm, height_mm))
}
