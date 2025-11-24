# Developed by Facundo Scordo, Research Assistance Professor @ UNR, Fall 2025; Email: scordo@agro.uba.ar

# Code to compare the results of the different model parametrization and the in-situ collections


# ===== Final R code: summary table + 2-panel figure (ordered by best model) =====

library(tidyverse)
library(cowplot)   # for side-by-side plotting

# 1) Load and pivot to wide
df <- read.csv("") # "Collection_vs_Model_Used_ERA5_Statistical_Comparison.csv" or "Collection_vs_Model_Used_WeatherStations_Statistical_Comparison.csv"
wide <- df %>%
  pivot_wider(names_from = Method, values_from = Value) %>%
  arrange(Site)

# 2) Stats function for one model vs COL
get_stats <- function(model_name, data) {
  obs <- data$COL
  mod <- data[[model_name]]
  
  keep <- is.finite(obs) & is.finite(mod)
  obs <- obs[keep]; mod <- mod[keep]
  
  fit <- lm(mod ~ obs)
  tibble(
    Model     = model_name,
    n         = length(obs),
    R         = cor(obs, mod),
    slope     = unname(coef(fit)[2]),
    intercept = unname(coef(fit)[1]),
    RMSE      = sqrt(mean((mod - obs)^2)),
    Bias      = mean(mod - obs)
  )
}

# 3) Build summary table for all models (exclude COL), then ORDER BY RMSE (best first)
models <- setdiff(names(wide), "COL")
summary_table <- purrr::map_dfr(models, get_stats, data = wide) %>%
  arrange(RMSE)   # <-- enforce best ??? worst order by RMSE

# Save the ordered table
write.csv(summary_table, "summary_table_ERA5.csv", row.names = FALSE)

# 4) Choose best model automatically (first row after ordering)
best_model <- summary_table$Model[1]

# 5) Panel (a): Average Bias (bars), color = RMSE, ordered by RMSE
p1 <- summary_table %>%
  mutate(Model = factor(Model, levels = Model)) %>%  # lock order from arrange(RMSE)
  ggplot(aes(x = Model, y = Bias, fill = RMSE)) +
  geom_col() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_viridis_c(option = "viridis", direction = -1) +  # darker = lower RMSE
  labs(
    title = "(a) Average Model Error vs. COL (Bias and RMSE)",
    x     = "Models",
    y     = "Average Bias across Sites (MOD - COL, g/m2)",
    fill  = "RMSE of MOD relative to COL\n (g m^-2 )"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 6) Panel (b): Observed vs Modeled scatter for the best model (circles + 1:1 line)
p2 <- ggplot(wide, aes(x = COL, y = .data[[best_model]])) +
  geom_point(color = "blue") +                     # circles
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = paste0("(b) Observed vs. Modeled Particle Mass (", best_model, ")"),
    x     = "COL (Observed, g m^-2)",
    y     = paste0("MOD (", best_model, ", g m^-2)")
  ) +
  theme_minimal(base_size = 12)

# 7) Combine & save
final_plot <- plot_grid(p1, p2, ncol = 2, rel_widths = c(1.3, 1))
ggsave("final_2panels_ERA5.png", final_plot, width = 12, height = 5, dpi = 300)
ggsave("final_2panels_ERA5.svg", final_plot, width = 12, height = 5)

