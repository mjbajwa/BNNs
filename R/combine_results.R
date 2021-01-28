# Combines results from different stan and fbm runs into comparative plots

# Load Libraries ----------------------------------------------------------

library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(janitor)
library(tidyr)
library(viridis)

# Configure paths ---------------------------------------------------------

PRIOR_ONLY <- F
OUTPUT_PATH <- "./output/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, "combined_", folder_name)
dir.create(path)

save_plot <- function(plot_object, name){
  
  ggsave(str_c(path, "/", name, ".png"),
         plot_object,
         width = 11,
         height = 8)
  
}

# Load stan and fbm objects -----------------------------------------------

if(PRIOR_ONLY){
  
  # Prior checks

  stan_centered_path <- "stan_2021_01_27_19_52_27" 
  stan_noncentered_path <- "stan_2021_01_27_19_54_41"
  fbm_path <- "fbm_2021_01_27_20_04_25"
  
} else {
  
  # Paths for complete runs - multi-chain etc.
    
  stan_centered_path <- "stan_2021_01_28_18_14_54" # stan_2021_01_27_20_10_37
  stan_noncentered_path <- "stan_2021_01_28_18_19_02" # stan_2021_01_27_20_13_51
  fbm_path <- "fbm_2021_01_28_18_12_40"  # fbm_2021_01_27_20_33_55
    
}

capture.output(list("centered" = stan_centered_path, 
                    "non-centered" = stan_noncentered_path,
                    "fbm" = fbm_path), 
               file = str_c(path, '/paths_used.txt'))

stan_centered <- read_rds(str_c("./output/", stan_centered_path, "/outputs.rds"))
stan_noncentered <- read_rds(str_c("./output/", stan_noncentered_path, "/outputs.rds"))
fbm <- read_rds(str_c("./output/", fbm_path, "/outputs.rds"))

# y vs x -------------------------------

# Extract and clean stan/hmc predictions

df_preds <- stan_centered$outputs$df_predictions %>% 
  mutate(method = "HMC: centered") %>% 
  bind_rows(stan_noncentered$outputs$df_predictions %>% mutate(method = "HMC: non-centered")) %>% 
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
  select(case, inputs, targets, mean, median, q1, q10, q90, q99, method)

# Extract fbm predictions

df_preds_fbm <- fbm$outputs$df_predictions %>% 
  mutate(method = "Gibbs") %>% 
  rename(mean = means,
         q10 = x10_qnt, 
         q90 = x90_qnt, 
         q1 = x1_qnt,
         q99 = x99_qnt) %>% 
  select(case, inputs, targets, mean, median, q1, q10, q90, q99, method)
  
df_preds <- df_preds %>% bind_rows(df_preds_fbm)

# Plot

df_train_inputs <- stan_centered$outputs$df_predictions %>%
  filter(label == "train") %>% 
  select(X_V1, mean) %>% 
  rename(inputs = X_V1)

y_vs_x_plot <- ggplot(df_preds) + 
  geom_ribbon(aes(x = inputs, ymin = q1, ymax = q99, fill = method), alpha = 0.1) + 
  geom_ribbon(aes(x = inputs, ymin = q10, ymax = q90, fill = method), alpha = 0.2) +
  geom_point(aes(x = inputs, y = targets), alpha = 0.5, color = "black", size = 2.5) +
  geom_line(aes(x = inputs, y = median, color = method), size = 0.8, alpha = 0.4) +
  geom_rug(data = df_train_inputs, aes(x = inputs, y = mean), sides="b") + 
  theme_bw() +
  xlab("X") +
  ylab("Y") +
  ggtitle(label = "Y vs. X (test set)",
          subtitle = "Shaded areas represent P10-P90 and P1-P99 intervals") + 
  theme(text = element_text(size = 20),
        legend.position = "bottom", 
        plot.subtitle = element_text(size = 12)) +
  facet_wrap(method ~ .)

y_vs_x_means <- ggplot() + 
  geom_point(data = df_preds, aes(x = inputs, y = targets), alpha = 0.5, color = "black", size = 2.5) +
  geom_line(data = df_preds, aes(x = inputs, y = median, color = method, linetype = method), size = 0.8, alpha = 0.8) +
  geom_rug(data = df_train_inputs, aes(x = inputs, y = mean), sides="b") + 
  theme_bw() +
  xlab("X") +
  ylab("Y") +
  ggtitle(label = "Y vs. X (test set)",
          subtitle = "Distribution of X in training data illustrated as rugs on the x-axis") + 
  theme(text = element_text(size = 20),
        legend.position = "bottom", 
        plot.subtitle = element_text(size = 12),
        legend.title = element_blank())

y_vs_x_plot %>% save_plot("y_vs_x")
y_vs_x_means %>% save_plot("y_vs_x_predictive_mean_only")

# Trace Plots -------------------------------------------------------------

markov_chain_samples <- function(stan_fit,
                                 var,
                                 n_chains = 4,
                                 burn_in = 1000,
                                 iters = 2000) {
  
  # Create empty dataframe to save results
  
  df_chains <- tibble()
  
  # Loop through each chain and extract the samples
  
  for (chain in 1:n_chains) {
    temp <- stan_fit@sim$samples[[chain]]
    parameter_samples <- temp[[var]]
    
    df_samples <- tibble(value = parameter_samples) %>% 
      mutate(t = 1:n(), 
             chain = chain)
    
    df_chains <- df_chains %>% 
      bind_rows(df_samples)
    
  }
  
  df_chains <- df_chains %>% mutate(var = var)
  
  return(df_chains)
  
}

join_fbm_stan_traces <- function(fbm_var, stan_var_pattern = "W\\[1", chosen_chain = 1, list_name = "desired_weight_vars"){
  
  # Extract FBM traces
  
  if(!is.null(fbm_var)){
  
    fbm_traces <- fbm$outputs$traces[[fbm_var]] %>% 
      mutate(method = "Gibbs") %>% 
      select(t, method, everything()) %>% 
      # filter(chain == chosen_chain) %>% 
      select(-chain) %>% 
      tidyr::pivot_longer(!c(t, method), names_to = "var")
  
  }
  
  # Extract stan centered traces
  
  stan_vars_c <- stan_centered$outputs[[list_name]][str_detect(stan_centered$outputs[[list_name]], stan_var_pattern)]
  
  stan_traces_c <- tibble()
  for(var in stan_vars_c){
    stan_traces_c <- stan_traces_c %>% 
      bind_rows(markov_chain_samples(stan_fit = stan_centered$outputs$stan_fit, var))
  }
  
  stan_traces_c <- stan_traces_c %>% 
    select(t, value, chain, var) %>% 
    tidyr::pivot_wider(names_from = var, values_from = value) %>% 
    mutate(method = "HMC: centered")
  
  # Extract stan uncentered traces
  
  stan_vars_nc <- stan_noncentered$outputs[[list_name]][str_detect(stan_noncentered$outputs[[list_name]], stan_var_pattern)]
  
  stan_traces_nc <- tibble()
  for(var in stan_vars_nc){
    stan_traces_nc <- stan_traces_nc %>% 
      bind_rows(markov_chain_samples(stan_fit = stan_noncentered$outputs$stan_fit, var))
  }
  
  stan_traces_nc <- stan_traces_nc %>% 
    select(t, value, chain, var) %>% 
    tidyr::pivot_wider(names_from = var, values_from = value) %>% 
    mutate(method = "HMC: non-centered")
  
  # Create one stan_traces data frame
  
  stan_traces <- stan_traces_c %>% 
    bind_rows(stan_traces_nc) %>% 
    # filter(chain == chosen_chain) %>% 
    select(-chain) %>% 
    select(t, method, everything()) %>% 
    tidyr::pivot_longer(!c(t, method), names_to = "var")
  
  # Join FBM and Stan traces
  
  if(!is.null(fbm_var)){
    all_traces <- stan_traces %>% 
      bind_rows(fbm_traces)
  } else {
    all_traces <- stan_traces
  }
  
  return(all_traces)
  
}

plot_traces <- function(df, title, subtext, size=0.2){
  
  ggplot(df) +
    geom_point(aes(x = t, y = value, color = var, alpha = t), size=size) + 
    geom_vline(xintercept = 1000, linetype = 2) + 
    theme_bw() + 
    viridis::scale_color_viridis(discrete=T) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=20),
          legend.position = "none", 
          plot.subtitle = element_text(size = 12)) + 
    facet_wrap(method ~ .) + 
    ggtitle(title, subtitle = subtext)
  
}

# Parameter Mapping

# Stan and FBM mapping

# w1 -> W[1, , ]
# w2 -> B[1, , ]
# w3 -> W[2, , ]
# w4 -> B[2, , ]
# h1 -> W_prec[1]
# h2 -> B_prec[2]
# h3 -> W_prec[2]

w1_traces <- join_fbm_stan_traces(fbm_var = "w1", stan_var_pattern = "W\\[1", chosen_chain = 1, list_name = "desired_weight_vars")
b1_traces <- join_fbm_stan_traces(fbm_var = "w2", stan_var_pattern = "B\\[1", chosen_chain = 1, list_name = "desired_bias_vars")
w2_traces <- join_fbm_stan_traces(fbm_var = "w3", stan_var_pattern = "W\\[2", chosen_chain = 1, list_name = "desired_weight_vars")
b2_traces <- join_fbm_stan_traces(fbm_var = "w4", stan_var_pattern = "B\\[2", chosen_chain = 1, list_name = "desired_bias_vars")
hw1_traces <- join_fbm_stan_traces(fbm_var = "h1", stan_var_pattern = "W_prec\\[1", chosen_chain = 1, list_name = "weights_desired_hp_vars")
hb1_traces <- join_fbm_stan_traces(fbm_var = "h2", stan_var_pattern = "B_prec\\[1", chosen_chain = 1, list_name = "biases_desired_hp_vars")
hw2_traces <- join_fbm_stan_traces(fbm_var = "h3", stan_var_pattern = "W_prec\\[2", chosen_chain = 1, list_name = "weights_desired_hp_vars")
y_prec_traces <- join_fbm_stan_traces(fbm_var = "y_sdev", stan_var_pattern = "y_prec", chosen_chain = 1, list_name = "target_noise_hp_vars")

SUBTEXT <- "Vertical line indicates starting point of values used as samples. Only one Stan chain is used for consistent comparison."

w1_trace_plot <- plot_traces(w1_traces, title = "Input-to-Hidden Weights", subtext = SUBTEXT)
b1_trace_plot <- plot_traces(b1_traces, title = "Hidden Unit Biases", subtext = SUBTEXT)
w2_trace_plot <- plot_traces(w2_traces, title = "Hidden-to-Output Unit Weights", subtext = SUBTEXT)
b2_trace_plot <- plot_traces(b2_traces, title = "Output Unit Biases", subtext = SUBTEXT)

hw1_trace_plot <- plot_traces(hw1_traces %>% mutate(value = ifelse(method == "Gibbs", 1/value^2, value)), 
            title = "Input-to-Hidden Weights: Precision Hyperparameter", subtext = SUBTEXT)

hb1_trace_plot <- plot_traces(hb1_traces %>% mutate(value = ifelse(method == "Gibbs", 1/value^2, value)), 
            title = "Hidden Unit Biases: Precision Hyperparameter", subtext = SUBTEXT)

hw2_trace_plot <- plot_traces(hw2_traces %>% mutate(value = ifelse(method == "Gibbs", 1/value^2, value)), 
            title = "Hidden-to-Output Weights: Precision Hyperparameter", subtext = SUBTEXT)

y_prec_trace_plot <- plot_traces(y_prec_traces %>% mutate(value = ifelse(method == "Gibbs", 1/value^2, value)), 
            title = "Target Noise: Precision Hyperparameter", 
            subtext = SUBTEXT, 
            size = 0.8)

# Save plots

w1_trace_plot %>% save_plot("input_to_hidden_weights")
b1_trace_plot %>% save_plot("hidden_unit_biases")
w2_trace_plot %>% save_plot("hidden_to_output_weights")
b2_trace_plot %>% save_plot("output_unit_bias")
hw1_trace_plot %>% save_plot("input_to_hidden_precision")
hb1_trace_plot %>% save_plot("hidden_bias_precision")
hw2_trace_plot %>% save_plot("hidden_to_output_precision")
y_prec_trace_plot %>% save_plot("target_noise_precision")

# Step size comparison  ---------------------------------------------------

if(PRIOR_ONLY == F){

  fbm_stepsizes <- fbm$outputs$df_chain_statistics %>% 
    ungroup() %>% 
    mutate(stepsize = stepsize*factor) %>% 
    rename(t = iteration) %>% 
    select(-factor) %>% 
    mutate(method = "Gibbs",
           chain = as.character(chain))
  
  stan_stepsizes_c <- stan_centered$outputs$df_chain_statistics %>% 
    select(iter, stepsize__, chain) %>% 
    rename(t = iter,
           stepsize = stepsize__) %>% 
    mutate(method = "HMC: centered", group = NA)
  
  stan_stepsizes_nc <- stan_noncentered$outputs$df_chain_statistics %>% 
    select(iter, stepsize__, chain) %>% 
    rename(t = iter,
           stepsize = stepsize__) %>% 
    mutate(method = "HMC: non-centered", group = NA)
  
  df_stepsizes <- fbm_stepsizes %>% 
    bind_rows(stan_stepsizes_c) %>% 
    bind_rows(stan_stepsizes_nc)
  
  stepsizes_plot <- ggplot(df_stepsizes %>% filter(t > 1000)) +
    geom_point(aes(x = t, y = stepsize, color = method), size = 0.5) + 
    theme_bw() + 
    facet_wrap(method~.) + 
    scale_color_viridis(discrete=T) + 
    xlab("") + 
    theme(text=element_text(size=20),
          legend.position = "none",
          plot.subtitle = element_text(size = 12)) + 
    facet_wrap(method ~ .) + 
    ggtitle("Stepsize comparison", 
            subtitle = "For HMC, all four chain's stepsizes are shown. For Gibbs, the different parameter group's step-sizes are shown.")
  
  stepsizes_plot %>% save_plot("step_size_comparison")

}

print(path)
