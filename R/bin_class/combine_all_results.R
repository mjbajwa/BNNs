# Load Libraries ----------------------------------------------------------

library(dplyr)
library(arrow)
library(stringr)
library(ggplot2)
library(tidyr)
library(readr)
library(gridExtra)
library(ROCit)

centered <- T
TRACE_THINNER <- 100

# Paths ------------------------------------------------------------

fbm_path <- "fbm_2021_05_04_18_04_39" # fbm_2021_04_27_15_25_14

if(centered){
  
  # Centered
  
  stan_path <- "stan_2021_04_27_13_15_11" # stan_2021_03_17_10_55_30 # stan_2021_04_27_13_15_11
  python_paths <- list(
    "NumPyro: NUTS" = "numpyro_c_2021_04_28_17_19_43" # numpyro_c_2021_04_27_14_28_38
    # "TFprobability: NUTS" = "tfprob_c_2021_03_23_11_31_12", 
    # "PyMC3: NUTS" = "pymc3_c_2021_03_23_12_40_08", 
    # "NumPyro: Gibbs-HMC" = "numpyro_c_2021_03_23_21_26_33"
  )
  
  # Additional parameters

  SUBTEXT <- "Centered Parametrization for all models excluding Gibbs/FBM."
  type <- "multiframework_c_"

} else {

  # Non-centered

  stan_path <- "stan_2021_04_29_11_21_32"  # stan_2021_03_23_17_33_49 stan_2021_03_17_12_30_29
  python_paths <- list(
    "NumPyro: NUTS" = "numpyro_nc_2021_04_29_12_41_38" 
    # "TFprobability: NUTS" = "tfprob_nc_2021_03_23_15_50_40", 
    # "PyMC3: NUTS" = "pymc3_nc_2021_03_23_13_41_24" 
  )
  
  SUBTEXT <- "Non-Centered Parametrization for all models excluding Gibbs/FBM."
  type <- "multiframework_nc_"
  
}


# Load data ---------------------------------------------------------------

# FBM

fbm <- read_rds(str_c("./output/bin_class/", fbm_path, "/outputs.rds"))

# Stan

stan <- read_rds(str_c("./output/bin_class/", stan_path, "/outputs.rds"))

# Python frameworks 

python <- list()

for(model in names(python_paths)){
  
  python_path <- python_paths[[model]]
  df_preds <- arrow::read_feather(str_c("./output/bin_class/", python_path, "/df_predictions.feather"))
  df_traces <- arrow::read_feather(str_c("./output/bin_class/", python_path, "/df_traces.feather"))
  
  # Dataframe adjustments
  
  df_preds <- df_preds %>% 
    mutate(method = model) %>% 
    rename(mean_prob = mean) %>% 
    select(X_V1, X_V2, targets, mean_prob, method, label)
  
  df_traces <- df_traces %>% 
    rename(chain = trace) %>% 
    mutate(method = model)
  
  python[[model]]$predictions <- df_preds
  python[[model]]$traces <- df_traces
  
}


# Predictions -------------------------------------------------------------

df_preds_fbm <- fbm$outputs$df_predictions %>% 
  mutate(method = "FBM: Gibbs-HMC") %>% 
  rename(targets = actual) %>% 
  select(X_V1, X_V2, targets, mean_prob, method) %>% 
  mutate(label = "test")

df_preds_stan <- stan$outputs$df_predictions %>% 
  mutate(method = "Stan") %>% 
  filter(label == "test") %>% 
  group_by(method) %>% 
  mutate(case = 1:n()) %>% 
  ungroup() %>% 
  rename(targets = actual) %>% 
  select(X_V1, X_V2, targets, mean_prob, method, label)

df_preds_all <- df_preds_fbm %>% 
  bind_rows(df_preds_stan)

for(i in names(python)){
  df_preds_all <- bind_rows(df_preds_all, python[[i]]$predictions %>% filter(label == "test"))
}

# Compute ROCs for each method

roc_test_stan <- rocit(score = df_preds_all %>% filter(label == "test", method == "Stan") %>% pull(mean_prob), 
                       class = df_preds_all %>% filter(label == "test", method == "Stan") %>% pull(targets))

roc_test_fbm <- rocit(score = df_preds_all %>% filter(label == "test", method == "FBM: Gibbs-HMC") %>% pull(mean_prob), 
                       class = df_preds_all %>% filter(label == "test", method == "FBM: Gibbs-HMC") %>% pull(targets))

roc_test_python <- rocit(score = df_preds_all %>% filter(label == "test", method == "NumPyro: NUTS") %>% pull(mean_prob), 
                        class = df_preds_all %>% filter(label == "test", method == "NumPyro: NUTS") %>% pull(targets))

df_tpr_fpr <- tibble(FPR = roc_test_stan$FPR, TPR = roc_test_stan$TPR, cutoff = roc_test_stan$Cutoff, method = "Stan") %>% 
  bind_rows(tibble(FPR = roc_test_fbm$FPR, TPR = roc_test_fbm$TPR, cutoff = roc_test_fbm$Cutoff, method = "FBM: Gibbs-HMC")) %>% 
  bind_rows(tibble(FPR = roc_test_python$FPR, TPR = roc_test_python$TPR, cutoff = roc_test_python$Cutoff, method = "NumPyro: NUTS"))

roc_plot <- ggplot(df_tpr_fpr) + 
  geom_line(aes(x = FPR, y = TPR, linetype = method, color = method), size = 2, alpha = 0.5) + 
  xlab("False Positive Rate - FPR") + 
  ylab("True Positive Rate - TPR") + 
  ggtitle("ROC curve for Test set", 
          subtitle = str_c("Stan AUC: ", round(roc_test_stan$AUC, 2), " | ", 
                           "FBM AUC: ", round(roc_test_fbm$AUC, 2), " | ", 
                           "NumPyro AUC: ", round(roc_test_python$AUC, 2))) + 
  theme_bw()

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
# y_prec_traces <- join_all_traces("y_sdev", "y_prec", "target_noise_hp_vars", "y_prec", sdev_conv = T)

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

# y_prec_trace_plot <- plot_traces(y_prec_traces, 
#                                  title = "Standard Deviation Hyperparameter: Target Noise", 
#                                  subtext = SUBTEXT, 
#                                  size = 0.5, log = F)

# Append to one PDF ---------------------------------------------------------

all_plots <- list(
  roc_plot,
  # y_prec_trace_plot,
  hw1_trace_plot,
  hw2_trace_plot,
  hb1_trace_plot,
  w1_trace_plot,
  b1_trace_plot,
  w2_trace_plot,
  b2_trace_plot
)


OUTPUT_PATH <- "./output/bin_class/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, type, folder_name)
dir.create(path)

ggsave(
  filename = str_c(path, "/", folder_name, "_", type, "_results.pdf"), 
  plot = marrangeGrob(all_plots, nrow=1, ncol=1), 
  width = 15, height = 9
)

print(path)

# Calculate misclassification error --------------------------------------------------------------------

# FBM

tmp_fbm <- fbm$outputs$df_predictions %>% 
  mutate(method = "FBM: Gibbs-HMC") %>% 
  rename(targets = actual) %>% 
  select(X_V1, X_V2, targets, predicted, mean_prob, method) %>% 
  mutate(label = "test")

fbm_misclass_error <- sum(abs(tmp_fbm$targets - tmp_fbm$predicted))/nrow(tmp_fbm)*100

# Stan

stan_prob_cutoff <- df_tpr_fpr %>% 
  filter(method == "Stan") %>% 
  filter(FPR <= 0.05) %>% 
  arrange(desc(FPR)) %>% 
  slice(1) %>% 
  pull(cutoff)

tmp_stan <- df_preds_stan %>% 
  mutate(predicted = ifelse(mean_prob > stan_prob_cutoff, 1, 0))

stan_misclass_error <- sum(abs(tmp_stan$targets - tmp_stan$predicted))/nrow(tmp_stan)*100

# Python

numpyro_prob_cutoff <- df_tpr_fpr %>% 
  filter(method == "NumPyro: NUTS") %>% 
  filter(FPR <= 0.05) %>% 
  arrange(desc(FPR)) %>% 
  slice(1) %>% 
  pull(cutoff)

tmp_numpyro <- df_preds_all %>% 
  filter(method == "NumPyro: NUTS") %>% 
  filter(label == "test") %>% 
  mutate(predicted = ifelse(mean_prob > numpyro_prob_cutoff, 1, 0))

numpyro_misclass_error <- sum(abs(tmp_numpyro$targets - tmp_numpyro$predicted))/nrow(tmp_numpyro)*100
  
print("Misclassification Errors")
print(str_c("fbm: ", fbm_misclass_error))
print(str_c("stan: ", stan_misclass_error))
print(str_c("numpyro: ", numpyro_misclass_error))

