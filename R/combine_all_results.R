# Load Libraries ----------------------------------------------------------

library(dplyr)
library(arrow)
library(stringr)
library(ggplot2)
library(tidyr)
library(readr)
library(gridExtra)

centered <- F
TRACE_THINNER <- 50

# Paths ------------------------------------------------------------

fbm_path <- "fbm_2021_03_17_20_06_09"

if(centered){
  
  # Centered
  
  stan_path <- "stan_2021_03_23_16_12_10" # stan_2021_03_17_10_55_30
  python_paths <- list(
    "NumPyro: NUTS" = "numpyro_c_2021_03_23_15_14_31", 
    "TFprobability: NUTS" = "tfprob_c_2021_03_23_11_31_12", 
    "PyMC3: NUTS" = "pymc3_c_2021_03_23_12_40_08", 
    "NumPyro: Gibbs-HMC" = "numpyro_c_2021_03_23_21_26_33"
  )
  
  # Additional parameters

  SUBTEXT <- "Centered Parametrization for all models excluding Gibbs/FBM."
  type <- "multiframework_c_"

} else {

  # Non-centered

  stan_path <- "stan_2021_03_23_17_33_49"  # stan_2021_03_23_17_33_49 stan_2021_03_17_12_30_29
  python_paths <- list(
    "NumPyro: NUTS" = "numpyro_nc_2021_03_24_09_45_54", 
    "TFprobability: NUTS" = "tfprob_nc_2021_03_23_15_50_40", 
    "PyMC3: NUTS" = "pymc3_nc_2021_03_23_13_41_24" 
  )
  
  SUBTEXT <- "Non-Centered Parametrization for all models excluding Gibbs/FBM."
  type <- "multiframework_nc_"
  
}

# Load data ---------------------------------------------------------------

# FBM

fbm <- read_rds(str_c("./output/", fbm_path, "/outputs.rds"))

# Stan

stan <- read_rds(str_c("./output/", stan_path, "/outputs.rds"))

# Python frameworks 

python <- list()

for(model in names(python_paths)){
  
  python_path <- python_paths[[model]]
  df_preds <- arrow::read_feather(str_c("./output/", python_path, "/df_predictions.feather"))
  df_traces <- arrow::read_feather(str_c("./output/", python_path, "/df_traces.feather"))
  
  # Dataframe adjustments
  
  df_preds <- df_preds %>% mutate(method = model)
  
  df_traces <- df_traces %>% 
    rename(chain = trace) %>% 
    mutate(method = model)
  
  python[[model]]$predictions <- df_preds
  python[[model]]$traces <- df_traces
  
}


# Predictions -------------------------------------------------------------

df_preds_fbm <- fbm$outputs$df_predictions %>% 
  mutate(method = "FBM: Gibbs-HMC") %>% 
  rename(mean = means,
         q10 = x10_qnt, 
         q90 = x90_qnt, 
         q1 = x1_qnt,
         q99 = x99_qnt) %>% 
  select(inputs, targets, mean, median, q1, q10, q90, q99, method) %>% 
  mutate(label = "test")

df_preds_stan <- stan$outputs$df_predictions %>% 
  mutate(method = "Stan") %>% 
  filter(label == "test") %>% 
  group_by(method) %>% 
  mutate(case = 1:n()) %>% 
  ungroup() %>% 
  rename(inputs = X_V1,
         targets = actual,
         median = `50%`,
         q10 = `10%`,
         q90 = `90%`,
         q1 = `1%`,
         q99 = `99%`
  ) %>% 
  select(inputs, targets, mean, median, q1, q10, q90, q99, method, label)

df_preds_all <- df_preds_fbm %>% 
  bind_rows(df_preds_stan)

for(i in names(python)){
  df_preds_all <- bind_rows(df_preds_all, python[[i]]$predictions)
}

df_train_inputs <- stan$outputs$df_predictions %>%
  filter(label == "train") %>% 
  select(X_V1, mean) %>% 
  rename(inputs = X_V1)

y_vs_x_plot <- ggplot(df_preds_all %>% filter(label == "test")) + 
  geom_ribbon(aes(x = inputs, ymin = q1, ymax = q99), fill = "gray2", alpha = 0.1) + 
  geom_ribbon(aes(x = inputs, ymin = q10, ymax = q90), fill = "gray2", alpha = 0.2) +
  geom_point(aes(x = inputs, y = targets), alpha = 0.9, color = "black", size = 2) +
  geom_line(aes(x = inputs, y = median), color = "gray4", size = 0.8, alpha = 0.6) +
  geom_rug(data = df_train_inputs, aes(x = inputs, y = mean), sides="b") + 
  theme_bw() +
  xlab("X") +
  ylab("Y") +
  ggtitle(label = "Y vs. X (test set)",
          subtitle = "Shaded areas represent P10-P90 and P1-P99 intervals") + 
  theme(text = element_text(size = 20),
        legend.position = "bottom", 
        plot.subtitle = element_text(size = 12)) +
  facet_wrap(method ~ ., dir = "h")


# Parameter Traces ------------------------------------------------------------------

plot_traces <- function(df, title, subtext, size = 0.15, thin = TRUE, log = FALSE){
  
  if(thin == TRUE){
    iters <- unique(df$t)
    keep_iters <- iters[seq(50, length(iters), TRACE_THINNER)]
  }
  
  final_plot <- ggplot(df %>% filter(t %in% keep_iters)) +
    geom_point(aes(x = t, y = value, color = factor(chain)), size=size) + 
    # geom_vline(xintercept = 1000, linetype = 2) + 
    theme_bw() + 
    scale_color_manual(values = c("red3", "blue3", "green4", "grey2"), name = "chain") + 
    # scale_color_grey(start = 0.05, end = 0.10) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=20),
          legend.position = "bottom", 
          plot.subtitle = element_text(size = 12), 
          axis.text.x = element_text(size = 10)) + 
    facet_wrap(method ~ .) + 
    ggtitle(title, subtitle = subtext) + 
    guides(colour = guide_legend(override.aes = list(size=6))) + 
    scale_x_continuous(label = function(x){format(x, scientific = TRUE)})
  
  if(log == TRUE){
    
    final_plot <- final_plot + 
      coord_trans(y = "log10")
    
  }
  
  return(final_plot)

}

# FBM

get_fbm_trace <- function(var){
  
  fbm$outputs$traces[[var]] %>% 
    mutate(method = "FBM: Gibbs-HMC") %>% 
    select(t, method, everything()) %>% 
    tidyr::pivot_longer(!c(t, method, chain), names_to = "name")
  
}

# Stan

get_single_stan_trace <- function(var,
                                  n_chains = 4,
                                  burn_in = 1000,
                                  iters = 2000) {
  
  # Create empty dataframe to save results
  
  df_chains <- tibble()
  
  # Loop through each chain and extract the samples
  
  for (chain in 1:n_chains) {
    temp <- stan$outputs$stan_fit@sim$samples[[chain]]
    parameter_samples <- temp[[var]]
    
    df_samples <- tibble(value = parameter_samples) %>%
      mutate(t = 1:n(),
             chain = chain)
    
    df_chains <- df_chains %>%
      bind_rows(df_samples)
    
  }
  
  df_chains %>% mutate(var = var)
  
}

get_stan_trace <- function(var, list_name, sdev_conv = FALSE){

  stan_vars <- stan$outputs[[list_name]][str_detect(stan$outputs[[list_name]], var)]
  
  stan_traces <- tibble()
  
  for(var in stan_vars){
    stan_traces <- stan_traces %>% 
      bind_rows(get_single_stan_trace(var))
  }
  
  stan_traces <- stan_traces %>% 
    select(t, value, chain, var) %>% 
    tidyr::pivot_wider(names_from = var, values_from = value) %>% 
    mutate(method = "Stan") %>% 
    select(t, method, everything()) %>% 
    tidyr::pivot_longer(!c(t, method, chain), names_to = "name")
  
  if(sdev_conv){
    stan_traces <- stan_traces %>% mutate(value = 1/sqrt(value))
  }
  
  return(stan_traces)
  
}

# Python frameworks

get_all_python_traces <- function(var, sdev_conv = FALSE){
  
  #' Returns traces for all key python plots
  
  df_traces <- tibble()

  for(model in names(python)){
  
    df_subset <- python[[model]]$traces %>% 
      select(contains(var), chain, method) %>% 
      group_by(chain) %>% 
      mutate(t = 1:n()) %>% 
      ungroup() %>% 
      select(t, everything()) %>% 
      pivot_longer(!c(t, chain, method))
  
    if(sdev_conv){
      df_subset <- df_subset %>% mutate(value = 1/sqrt(value))
    }
    
    df_traces <- bind_rows(df_traces, df_subset)
    
  }
  
  return(df_traces)

}

# Join all traces

join_all_traces <- function(fbm_var, stan_var, stan_list, python_var, sdev_conv=FALSE){
  
  #' Returns the joined traces for plotting purposes
  
  df_combined_traces <- get_fbm_trace(fbm_var) %>% 
    bind_rows(get_stan_trace(stan_var, stan_list, sdev_conv)) %>% 
    bind_rows(get_all_python_traces(python_var, sdev_conv))
  
  return(df_combined_traces)
  
}

# fbm_var, stan_var, stan_list, python_var, py_sdev_conv

w1_traces <- join_all_traces("w1", "W\\[1", "desired_weight_vars", "w_ih") 
b1_traces <- join_all_traces("w2", "B\\[1", "desired_bias_vars", "b_h") 
w2_traces <- join_all_traces("w3", "W\\[2", "desired_weight_vars", "w_ho")
b2_traces <- join_all_traces("w4", "B\\[1,2", "desired_bias_vars", "b_o")
hw1_traces <- join_all_traces("h1", "W_prec\\[1" , "weights_desired_hp_vars", "W_prec_ih", sdev_conv = T) 
hb1_traces <- join_all_traces("h2", "B_prec\\[1", "biases_desired_hp_vars", "B_prec_h", sdev_conv = T) 
hw2_traces <- join_all_traces("h3", "W_prec\\[2", "weights_desired_hp_vars", "W_prec_ho", sdev_conv = T)
y_prec_traces <- join_all_traces("y_sdev", "y_prec", "target_noise_hp_vars", "y_prec", sdev_conv = T)

# Plots

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
                                 size = 0.5, log = F)

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


OUTPUT_PATH <- "./output/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, type, folder_name)
dir.create(path)

ggsave(
  filename = str_c(path, "/", folder_name, "_", type, "_results.pdf"), 
  plot = marrangeGrob(all_plots, nrow=1, ncol=1), 
  width = 15, height = 9
)

print(path)
