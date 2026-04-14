# ============================================================
# Lil Uzi Vert - Spotify Streaming EDA
# ============================================================
# Requirements: install.packages(c("tidyverse", "scales", "corrplot"))

library(tidyverse)
library(scales)
library(corrplot)

# ── 1. LOAD & CLEAN ─────────────────────────────────────────

# Update this path to wherever your CSV lives
df <- read_csv("uzi_songs.csv")

# The asterisk (*) prefix on song_title marks feature tracks
# Pull that into its own boolean column, then strip it from the title
df <- df %>%
  mutate(
    is_feature = str_starts(song_title, "\\*"),
    song_title = str_remove(song_title, "^\\*"),
    # Shorten long titles for plotting (cap at 35 chars)
    short_title = ifelse(
      nchar(song_title) > 35,
      paste0(str_sub(song_title, 1, 32), "..."),
      song_title
    ),
    # Streams in millions / thousands for readability
    streams_m = Streams / 1e6,
    daily_k   = Daily / 1e3
  )

cat("Dataset dimensions:", nrow(df), "songs x", ncol(df), "columns\n\n")


# ── 2. SUMMARY STATISTICS ───────────────────────────────────

# Helper: statistical mode (most frequent value)
stat_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

cat("═══════════════════════════════════════════\n")
cat("       TOTAL STREAMS (raw counts)\n")
cat("═══════════════════════════════════════════\n")
cat("  Mean:    ", format(mean(df$Streams), big.mark = ","), "\n")
cat("  Median:  ", format(median(df$Streams), big.mark = ","), "\n")
cat("  Mode:    ", format(stat_mode(df$Streams), big.mark = ","), "\n")
cat("  Std Dev: ", format(sd(df$Streams), big.mark = ","), "\n")
cat("  Min:     ", format(min(df$Streams), big.mark = ","), "\n")
cat("  Max:     ", format(max(df$Streams), big.mark = ","), "\n")
cat("  IQR:     ", format(IQR(df$Streams), big.mark = ","), "\n")
cat("\n")

cat("═══════════════════════════════════════════\n")
cat("       DAILY STREAMS (raw counts)\n")
cat("═══════════════════════════════════════════\n")
cat("  Mean:    ", format(mean(df$Daily), big.mark = ","), "\n")
cat("  Median:  ", format(median(df$Daily), big.mark = ","), "\n")
cat("  Mode:    ", format(stat_mode(df$Daily), big.mark = ","), "\n")
cat("  Std Dev: ", format(sd(df$Daily), big.mark = ","), "\n")
cat("  Min:     ", format(min(df$Daily), big.mark = ","), "\n")
cat("  Max:     ", format(max(df$Daily), big.mark = ","), "\n\n")

# Lead vs Feature breakdown
cat("═══════════════════════════════════════════\n")
cat("       LEAD vs FEATURE BREAKDOWN\n")
cat("═══════════════════════════════════════════\n")
df %>%
  group_by(is_feature) %>%
  summarise(
    n_tracks      = n(),
    avg_streams   = mean(Streams),
    median_streams = median(Streams),
    avg_daily     = mean(Daily),
    .groups = "drop"
  ) %>%
  mutate(role = ifelse(is_feature, "Feature", "Lead")) %>%
  select(role, everything(), -is_feature) %>%
  print()
cat("\n")


# ── 3. PLOT: TOP 15 SONGS BY TOTAL STREAMS ──────────────────

top15 <- df %>%
  slice_max(Streams, n = 15)

p1 <- ggplot(top15, aes(x = reorder(short_title, streams_m), y = streams_m)) +
  geom_col(aes(fill = is_feature), width = 0.7, show.legend = TRUE) +
  geom_text(
    aes(label = paste0(round(streams_m, 0), "M")),
    hjust = -0.1, size = 3.2, color = "grey30"
  ) +
  coord_flip() +
  scale_fill_manual(
    values = c("FALSE" = "#7B2FF7", "TRUE" = "#FF4365"),
    labels = c("Lead Artist", "Feature")
  ) +
  scale_y_continuous(
    labels = label_comma(suffix = "M"),
    expand = expansion(mult = c(0, 0.15))
  ) +
  labs(
    title    = "Lil Uzi Vert — Top 15 Songs by Total Streams",
    subtitle = "Source: kworb.net Spotify data",
    x = NULL,
    y = "Total Streams (millions)",
    fill = "Role"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey50"),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom"
  )

print(p1)
ggsave("plot_top15_total_streams.png", p1, width = 10, height = 7, dpi = 150)
cat("Saved: plot_top15_total_streams.png\n")


# ── 4. PLOT: TOP 10 SONGS BY DAILY STREAMS ──────────────────

top10_daily <- df %>%
  slice_max(Daily, n = 10)

p2 <- ggplot(top10_daily, aes(x = reorder(short_title, daily_k), y = daily_k)) +
  geom_col(aes(fill = is_feature), width = 0.7, show.legend = TRUE) +
  geom_text(
    aes(label = paste0(round(daily_k, 1), "K")),
    hjust = -0.1, size = 3.2, color = "grey30"
  ) +
  coord_flip() +
  scale_fill_manual(
    values = c("FALSE" = "#00D4AA", "TRUE" = "#FF6B35"),
    labels = c("Lead Artist", "Feature")
  ) +
  scale_y_continuous(
    labels = label_comma(suffix = "K"),
    expand = expansion(mult = c(0, 0.15))
  ) +
  labs(
    title    = "Lil Uzi Vert — Top 10 Songs by Daily Streams",
    subtitle = "Current daily streaming velocity",
    x = NULL,
    y = "Daily Streams (thousands)",
    fill = "Role"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey50"),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom"
  )

print(p2)
ggsave("plot_top10_daily_streams.png", p2, width = 10, height = 6, dpi = 150)
cat("Saved: plot_top10_daily_streams.png\n")


# ── 5. SCATTERPLOT: TOTAL vs DAILY STREAMS ──────────────────

p3 <- ggplot(df, aes(x = streams_m, y = daily_k)) +
  geom_point(
    aes(color = is_feature),
    alpha = 0.6, size = 2.5
  ) +
  # Label outliers (top 5 by daily OR top 5 by total)
  geom_text(
    data = df %>%
      filter(
        rank(-Daily) <= 5 | rank(-Streams) <= 5
      ),
    aes(label = short_title),
    size = 2.5, hjust = -0.1, vjust = -0.5,
    check_overlap = TRUE, color = "grey20"
  ) +
  # Add a trend line
  geom_smooth(method = "lm", se = TRUE, color = "#7B2FF7",
              linewidth = 0.8, alpha = 0.15) +
  scale_color_manual(
    values = c("FALSE" = "#7B2FF7", "TRUE" = "#FF4365"),
    labels = c("Lead Artist", "Feature")
  ) +
  scale_x_continuous(labels = label_comma(suffix = "M")) +
  scale_y_continuous(labels = label_comma(suffix = "K")) +
  labs(
    title    = "Total Streams vs. Daily Streams",
    subtitle = "Does catalog size predict current momentum?",
    x = "Total Streams (millions)",
    y = "Daily Streams (thousands)",
    color = "Role"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey50"),
    legend.position = "bottom"
  )

print(p3)
ggsave("plot_scatter_total_vs_daily.png", p3, width = 10, height = 7, dpi = 150)
cat("Saved: plot_scatter_total_vs_daily.png\n")


# ── 6. CORRELATION ANALYSIS ─────────────────────────────────

# Pearson correlation between Streams and Daily
r_pearson  <- cor(df$Streams, df$Daily, method = "pearson")
r_spearman <- cor(df$Streams, df$Daily, method = "spearman")

cat("\n═══════════════════════════════════════════\n")
cat("       CORRELATION: Streams vs Daily\n")
cat("═══════════════════════════════════════════\n")
cat("  Pearson r:  ", round(r_pearson, 4), "\n")
cat("  Spearman ρ: ", round(r_spearman, 4), "\n")

# Significance test
cor_test <- cor.test(df$Streams, df$Daily)
cat("  p-value:    ", format.pval(cor_test$p.value, digits = 4), "\n")
cat("  95% CI:     [", round(cor_test$conf.int[1], 4), ",",
    round(cor_test$conf.int[2], 4), "]\n\n")

# Build a small correlation matrix including the numeric + derived cols
cor_df <- df %>%
  mutate(is_feature_num = as.numeric(is_feature)) %>%
  select(Streams, Daily, is_feature_num)

cor_matrix <- cor(cor_df, use = "complete.obs")

cat("Correlation Matrix:\n")
print(round(cor_matrix, 4))

# Visual correlation matrix
png("plot_correlation_matrix.png", width = 600, height = 500, res = 120)
corrplot(
  cor_matrix,
  method   = "color",
  type     = "upper",
  addCoef.col = "black",
  tl.col   = "black",
  tl.srt   = 45,
  col      = colorRampPalette(c("#FF4365", "white", "#7B2FF7"))(200),
  title    = "Correlation Matrix",
  mar      = c(0, 0, 2, 0)
)
dev.off()
cat("Saved: plot_correlation_matrix.png\n")

