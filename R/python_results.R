# Load Libraries ----------------------------------------------------------

library(dplyr)
library(arrow)
library(stringr)
library(ggplot2)
library(tidyr)
library(readr)
library(gridExtra)
  
# User Inputs ------------------------------------------------------------

METHOD <- "NumPyro (Gibbs-HMC)"
python_path <- "numpyro_c_2021_04_02_12_23_28" #  numpyro_nc_2021_03_19_11_33_38 numpyro_c_2021_03_21_15_46_28
fbm_path <- "fbm_2021_03_19_07_55_29"

# 500k runs

# numpyro_nc_2021_03_19_09_08_57
# fbm_2021_03_19_07_55_29

# 500k

# numpyro_nc_2021_03_22_13_10_12

# Output path generation ------------------------------------------------

OUTPUT_PATH <- "./output/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, "python_combined_", folder_name)
dir.create(path)

python_path <- str_c("./output/", python_path)

capture.output(list("python" = python_path, 
                    "fbm" = fbm_path), 
               file = str_c(path, '/paths_used.txt'))

# Load FBM ----------------------------------------------------------------

fbm <- read_rds(str_c("./output/", fbm_path, "/outputs.rds"))

df_preds_fbm <- fbm$outputs$df_predictions %>% 
  mutate(method = "Gibbs") %>% 
  rename(mean = means,
         q10 = x10_qnt, 
         q90 = x90_qnt, 
         q1 = x1_qnt,
         q99 = x99_qnt) %>% 
  select(inputs, targets, mean, median, q1, q10, q90, q99, method) %>% 
  mutate(label = "test")

# Load Results ---------------------------------------------------------------

df_preds <- arrow::read_feather(str_c(python_path, "/df_predictions.feather"))
df_traces <- arrow::read_feather(str_c(python_path, "/df_traces.feather"))

df_traces <- df_traces %>% 
  rename(chain = trace) %>% 
  mutate(method = METHOD)

# Prediction Plot ---------------------------------------------------------

df_preds <- df_preds %>% mutate(method = METHOD)
df_preds <- df_preds %>% bind_rows(df_preds_fbm)

y_vs_x_plot <- ggplot(df_preds %>% filter(label == "test")) + 
  geom_ribbon(aes(x = inputs, ymin = q1, ymax = q99), fill = "gray2", alpha = 0.1) + 
  geom_ribbon(aes(x = inputs, ymin = q10, ymax = q90), fill = "gray2", alpha = 0.2) +
  geom_point(aes(x = inputs, y = targets), alpha = 0.9, color = "black", size = 2) +
  geom_line(aes(x = inputs, y = median), color = "gray4", size = 0.8, alpha = 0.6) +
  geom_rug(data = df_preds %>% filter(label == "train"), aes(x = inputs, y = mean), sides="b") + 
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

plot_traces <- function(df, title, subtext, size = 0.15, thin = TRUE, log = FALSE){
  
  if(thin == TRUE){
    iters <- unique(df$t)
    keep_iters <- iters[seq(1, length(iters), 100)]
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
          plot.subtitle = element_text(size = 12), 
          axis.text.x = element_text(size = 10)) + 
    facet_wrap(method ~ ., scales = "free_x") + 
    ggtitle(title, subtitle = subtext) + 
    guides(colour = guide_legend(override.aes = list(size=6))) + 
    scale_x_continuous(label = function(x){format(x, scientific = TRUE)})
  
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
                "w_ih" = "Input-to-Hidden Weights", 
                "w_ho" = "Hidden Unit Biases", 
                "b_h" = "Hidden-to-Output Unit Weights", 
                "b_o" = "Output Unit Biases")

join_fbm_other_traces <- function(fbm_var, other_var, sdev_conversion = F){
  
  # Extract FBM traces
  
  if(!is.null(fbm_var)){
    
    fbm_traces <- fbm$outputs$traces[[fbm_var]] %>% 
      mutate(method = "Gibbs") %>% 
      select(t, method, everything()) %>% 
      tidyr::pivot_longer(!c(t, method, chain), names_to = "name")
  }
  
  # Other
  
  df_subset <- df_traces %>% 
    select(contains(other_var), chain, method) %>% 
    group_by(chain) %>% 
    mutate(t = 1:n()) %>% 
    ungroup() %>% 
    select(t, everything()) %>% 
    pivot_longer(!c(t, chain, method))
  
  if(sdev_conversion){
    df_subset <- df_subset %>% mutate(value = 1/sqrt(value))
  }
  
  fbm_traces %>% bind_rows(df_subset)
  
}

# Create plots and save results -------------------------------------------

SUBTEXT <- ""

w1_traces <- join_fbm_other_traces(fbm_var = "w1", other_var = "w_ih") 
b1_traces <- join_fbm_other_traces(fbm_var = "w2", other_var = "b_h") 
w2_traces <- join_fbm_other_traces(fbm_var = "w3", other_var = "w_ho")
b2_traces <- join_fbm_other_traces(fbm_var = "w4", other_var = "b_o")
hw1_traces <- join_fbm_other_traces(fbm_var = "h1", other_var = "W_prec_ih", sdev_conversion = T) 
hb1_traces <- join_fbm_other_traces(fbm_var = "h2", other_var = "B_prec_h", sdev_conversion = T) 
hw2_traces <- join_fbm_other_traces(fbm_var = "h3", other_var = "W_prec_ho", sdev_conversion = T)
y_prec_traces <- join_fbm_other_traces(fbm_var = "y_sdev", other_var = "y_prec", sdev_conversion = T)

w1_trace_plot <- plot_traces(w1_traces, title = "Input-to-Hidden Weights", subtext = SUBTEXT)
b1_trace_plot <- plot_traces(b1_traces, title = "Hidden Unit Biases", subtext = SUBTEXT)
w2_trace_plot <- plot_traces(w2_traces, title = "Hidden-to-Output Unit Weights", subtext = SUBTEXT)
b2_trace_plot <- plot_traces(b2_traces, title = "Output Unit Biases", subtext = SUBTEXT)

hw1_trace_plot <- plot_traces(hw1_traces, 
                              title = "Standard Deviation Hyperparameter: Input-to-Hidden Weights", subtext = SUBTEXT, log = T)

hb1_trace_plot <- plot_traces(hb1_traces, 
                              title = "Standard Deviation Hyperparameter: Hidden Unit Biases", subtext = SUBTEXT, log = T)

hw2_trace_plot <- plot_traces(hw2_traces,
                              title = "Standard Deviation Hyperparameter: Hidden-to-Output Weights", subtext = SUBTEXT, log = T)

y_prec_trace_plot <- plot_traces(y_prec_traces, 
                                 title = "Standard Deviation Hyperparameter: Target Noise", 
                                 subtext = SUBTEXT, 
                                 size = 0.5, log = T)

# Save all results to disk ------------------------------------------------

ggsave(
  str_c(path, "/predicted_vs_actual.png"),
  y_vs_x_plot,
  width = 11,
  height = 8
)

save_plot <- function(plot_object, name){
  
  ggsave(str_c(path, "/", name, ".png"),
         plot_object,
         width = 11,
         height = 8)
  
}

w1_trace_plot %>% save_plot("input_to_hidden_weights")
b1_trace_plot %>% save_plot("hidden_unit_biases")
w2_trace_plot %>% save_plot("hidden_to_output_weights")
b2_trace_plot %>% save_plot("output_unit_bias")
hw1_trace_plot %>% save_plot("input_to_hidden_precision")
hb1_trace_plot %>% save_plot("hidden_bias_precision")
hw2_trace_plot %>% save_plot("hidden_to_output_precision")
y_prec_trace_plot %>% save_plot("target_noise_precision")

# Append to one PDF ---------------------------------------------------------

all_plots <- list(
  y_vs_x_plot,
  y_prec_trace_plot,
  hw1_trace_plot,
  hw2_trace_plot,
  hb1_trace_plot,
  w1_trace_plot,
  b1_trace_plot,
  w2_trace_plot,
  b2_trace_plot
)

ggsave(
  filename = str_c(path, "/", folder_name, "_combined_results.pdf"), 
  plot = marrangeGrob(all_plots, nrow=1, ncol=1), 
  width = 15, height = 9
)

print(path)

