# Load Libraries ----------------------------------------------------------

library(dplyr)
library(arrow)
library(stringr)
library(ggplot2)
library(tidyr)

# Centered = pymc3_2021_02_21_10_16_01
# Non-centered = pymc3_2021_02_21_11_07_40

PYMC3_PATH <- "./output/pymc3_2021_02_21_11_07_40/"

# Load Data ---------------------------------------------------------------

df_predictions <- arrow::read_feather(str_c(PYMC3_PATH, "df_predictions.feather"))
df_traces <- arrow::read_feather(str_c(PYMC3_PATH, "df_traces.feather"))

df_traces <- df_traces %>% 
  rename(chain = trace) %>% 
  mutate(method = "PyMC3")

# Prediction Plot ---------------------------------------------------------

df_predictions <- df_predictions %>% mutate(method = "NUTS (Centered): PyMC3")

y_vs_x_plot <- ggplot(df_predictions %>% filter(label == "test")) + 
  geom_ribbon(aes(x = inputs, ymin = q1, ymax = q99), fill = "gray2", alpha = 0.1) + 
  geom_ribbon(aes(x = inputs, ymin = q10, ymax = q90), fill = "gray2", alpha = 0.2) +
  geom_point(aes(x = inputs, y = targets), alpha = 0.9, color = "black", size = 2) +
  geom_line(aes(x = inputs, y = median), color = "gray4", size = 0.8, alpha = 0.6) +
  geom_rug(data = df_predictions %>% filter(label == "train"), aes(x = inputs, y = mean), sides="b") + 
  theme_bw() +
  xlab("X") +
  ylab("Y") +
  ggtitle(label = "Y vs. X (test set)",
          subtitle = "Shaded areas represent P10-P90 and P1-P99 intervals") + 
  theme(text = element_text(size = 20),
        legend.position = "bottom", 
        plot.subtitle = element_text(size = 12)) +
  facet_wrap(method ~ .)

# Trace Plots -------------------------------------------------------------

plot_traces <- function(df, var_id, title, subtext, size = 0.15, thin = TRUE, log = FALSE){
  
  if(thin == TRUE){
    iters <- unique(df$t)
    keep_iters <- iters[seq(1, length(iters), 10)]
  }
  
  final_plot <- ggplot(df %>% filter(t %in% keep_iters)) +
    geom_point(aes(x = t, y = value, color = factor(chain)), size=size) + 
    geom_vline(xintercept = 1000, linetype = 2) + 
    theme_bw() + 
    scale_color_manual(values = c("red3", "blue3", "green4", "grey2"), name = "chain") + 
    # scale_color_grey(start = 0.05, end = 0.10) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=20),
          legend.position = "bottom", 
          plot.subtitle = element_text(size = 12)) + 
    facet_wrap(method ~ .) + 
    ggtitle(title, subtitle = subtext) + 
    guides(colour = guide_legend(override.aes = list(size=6)))
  
  if(log == TRUE){
    
    final_plot <- final_plot + 
      coord_trans(y = "log10")
    
  }
  
  return(final_plot)
}

var_ids <- list("W_prec_ih" = "Standard Deviation Hyperparameter: Input-to-Hidden Weights", 
                "W_prec_ho" = "Standard Deviation Hyperparameter: Hidden-to-Output Weights", 
                "B_prec_h" = "Standard Deviation Hyperparameter: Hidden Unit Biases", 
                "y_prec" = "Standard Deviation Hyperparameter: Target Noise", 
                "w_ih_raw" = "Input-to-Hidden Weights", 
                "w_ho_raw" = "Hidden Unit Biases", 
                "b_h_raw" = "Hidden-to-Output Unit Weights", 
                "b_o" = "Output Unit Biases")

trace_plots <- list()

for(var_id in names(var_ids)){
  
  print(var_id)
  
  df_subset <- df_traces %>% 
    select(contains(var_id), chain, method) %>% 
    group_by(chain) %>% 
    mutate(t = 1:n()) %>% 
    ungroup() %>% 
    select(t, everything()) %>% 
    pivot_longer(!c(t, chain, method))
  
  if(str_detect(var_id, "prec")){
    df_subset <- df_subset %>% mutate(value = 1/sqrt(value))
  }
   
  # if(str_detect(var_id, "W_prec_ho")){
  #   df_subset <- df_subset %>% mutate(value = value * 1/sqrt(8))
  # 
  # }
  
  trace_plots[[var_id]] <- plot_traces(df_subset, var_id, title = var_ids[var_id], subtext = "")
  
}

# Save all results to disk ------------------------------------------------

ggsave(
  str_c(PYMC3_PATH, "/predicted_vs_actual.png"),
  y_vs_x_plot,
  width = 11,
  height = 8
)

for(i in 1:length(trace_plots)){
  
  ggsave(
    str_c(PYMC3_PATH, "/", names(trace_plots)[i], ".png"),
    trace_plots[[1]],
    width = 11,
    height = 8
  )
  
}
