# Load Libraries ----------------------------------------------------------

library(dplyr)
library(arrow)
library(stringr)
library(ggplot2)
library(tidyr)
library(readr)
library(gridExtra)
library(ROCit)

centered <- F
TRACE_THINNER <- 100

# Paths ------------------------------------------------------------

# fbm_path <- "fbm_2021_05_07_12_02_31" # fbm_2021_04_27_15_25_14 fbm_2021_05_05_10_48_54 fbm_2021_05_13_18_57_55
fbm_path <- "fbm_2021_05_20_11_28_28"

if(centered){
  
  # Centered
  
  stan_path <- "stan_2021_05_05_20_36_09" # stan_2021_03_17_10_55_30 # stan_2021_04_27_13_15_11
  python_paths <- list(
    "NumPyro: NUTS" = "numpyro_c_2021_05_06_14_01_52" # numpyro_c_2021_04_27_14_28_38 # 100k: numpyro_c_2021_05_13_11_28_25
    # "TFprobability: NUTS" = "tfprob_c_2021_03_23_11_31_12", 
    # "PyMC3: NUTS" = "pymc3_c_2021_03_23_12_40_08", 
    # "NumPyro: Gibbs-HMC" = "numpyro_c_2021_03_23_21_26_33"
  )
  
  # Additional parameters

  SUBTEXT <- "Centered Parametrization for all models excluding Gibbs/FBM."
  type <- "multiframework_c_"

} else {

  # Non-centered

  stan_path <- "stan_2021_05_07_16_23_31"  # stan_2021_03_23_17_33_49 stan_2021_03_17_12_30_29
  python_paths <- list(
    "NumPyro: NUTS - non-centered" = "numpyro_nc_2021_05_13_13_01_48",  # 2k runs: numpyro_nc_2021_05_06_19_31_07
    "NumPyro: NUTS - centered" = "numpyro_c_2021_05_13_11_28_25" # numpyro_nc_2021_05_13_13_01_48
    # "TFprobability: NUTS" = "tfprob_nc_2021_03_23_15_50_40", 
    # "PyMC3: NUTS" = "pymc3_nc_2021_03_23_13_41_24" 
  )
  
  SUBTEXT <- "" # Non-Centered Parametrization for all models excluding Gibbs/FBM."
  type <- "multiframework_nc_"
  
}


# Load data ---------------------------------------------------------------

# FBM

fbm <- read_rds(str_c("./output/multi_class/", fbm_path, "/outputs.rds"))

# Stan

stan <- read_rds(str_c("./output/multi_class/", stan_path, "/outputs.rds"))

# Python frameworks 

python <- list()

for(model in names(python_paths)){
  
  python_path <- python_paths[[model]]
  df_preds <- arrow::read_feather(str_c("./output/multi_class/", python_path, "/df_predictions.feather"))
  df_traces <- arrow::read_feather(str_c("./output/multi_class/", python_path, "/df_traces.feather"))
  
  # Dataframe adjustments
  
  df_preds <- df_preds %>% 
    mutate(method = model) %>% 
    select(contains("X_"), targets, predicted, contains("mean_prob"), method, label)
  
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
  select(contains("X_"), targets, predicted, contains("mean_prob"), method) %>% 
  mutate(label = "test")

df_preds_stan <- stan$outputs$df_predictions %>% 
  mutate(method = "Stan") %>% 
  filter(label == "test") %>% 
  group_by(method) %>% 
  mutate(case = 1:n()) %>% 
  ungroup() %>% 
  rename(targets = actual) %>% 
  select(case, contains("X_"), targets, predicted, method, label) %>% 
  left_join(stan$outputs$df_prob %>% filter(label == "test") %>% select(-label), by = c("case" = "index")) %>% 
  select(-case)

df_preds_all <- df_preds_fbm %>% 
  bind_rows(df_preds_stan)

for(i in names(python)){
  df_preds_all <- bind_rows(df_preds_all, python[[i]]$predictions %>% filter(label == "test"))
}

# # Compute ROCs for each method
# 
# roc_test_stan <- rocit(score = df_preds_all %>% filter(label == "test", method == "Stan") %>% pull(mean_prob), 
#                        class = df_preds_all %>% filter(label == "test", method == "Stan") %>% pull(targets))
# 
# roc_test_fbm <- rocit(score = df_preds_all %>% filter(label == "test", method == "FBM: Gibbs-HMC") %>% pull(mean_prob), 
#                        class = df_preds_all %>% filter(label == "test", method == "FBM: Gibbs-HMC") %>% pull(targets))
# 
# roc_test_python <- rocit(score = df_preds_all %>% filter(label == "test", method == "NumPyro: NUTS") %>% pull(mean_prob), 
#                         class = df_preds_all %>% filter(label == "test", method == "NumPyro: NUTS") %>% pull(targets))
# 
# df_tpr_fpr <- tibble(FPR = roc_test_stan$FPR, TPR = roc_test_stan$TPR, cutoff = roc_test_stan$Cutoff, method = "Stan") %>% 
#   bind_rows(tibble(FPR = roc_test_fbm$FPR, TPR = roc_test_fbm$TPR, cutoff = roc_test_fbm$Cutoff, method = "FBM: Gibbs-HMC")) %>% 
#   bind_rows(tibble(FPR = roc_test_python$FPR, TPR = roc_test_python$TPR, cutoff = roc_test_python$Cutoff, method = "NumPyro: NUTS"))
# 
# roc_plot <- ggplot(df_tpr_fpr) + 
#   geom_line(aes(x = FPR, y = TPR, linetype = method, color = method), size = 2, alpha = 0.5) + 
#   xlab("False Positive Rate - FPR") + 
#   ylab("True Positive Rate - TPR") + 
#   ggtitle("ROC curve for Test set", 
#           subtitle = str_c("Stan AUC: ", round(roc_test_stan$AUC, 2), " | ", 
#                            "FBM AUC: ", round(roc_test_fbm$AUC, 2), " | ", 
#                            "NumPyro AUC: ", round(roc_test_python$AUC, 2))) + 
#   theme_bw()

# Parameter Traces ------------------------------------------------------------------

plot_traces <- function(df, title, subtext, size = 0.15, thin = TRUE, log = FALSE, ard = FALSE){
  
  df <- df %>% filter(t <= 20000)
  
  if(thin == TRUE){
    iters <- unique(df$t)
    keep_iters <- iters[seq(1, length(iters), TRACE_THINNER)]
  }
  
  if(ard){
    
    final_plot <- ggplot(df %>% filter(t %in% keep_iters)) +
      geom_point(aes(x = t, y = value, color = name), size=size) + 
      # geom_vline(xintercept = 1000, linetype = 2) + 
      theme_bw() + 
      # scale_color_manual(values = c("red3", "blue3", "green4", "grey2"), name = "chain") + 
      # scale_color_grey(start = 0.05, end = 0.10) + 
      xlab("") + 
      ylab("") + 
      theme(text=element_text(size=16),
            legend.position = "bottom", 
            plot.subtitle = element_text(size = 12), 
            axis.text.x = element_text(size = 10), 
            axis.text.y = element_text(size = 5)) + 
      # facet_wrap(method ~ .) + 
      facet_grid(chain ~ method) + 
      ggtitle(title, subtitle = subtext) + 
      guides(colour = guide_legend(override.aes = list(size=6))) + 
      scale_x_continuous(label = function(x){format(x, scientific = TRUE)})
  
    } else {
        
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
              axis.text.x = element_text(size = 10), 
              axis.text.y = element_text(size = 10)) + 
        facet_wrap(method ~ .) + 
        ggtitle(title, subtitle = subtext) + 
        guides(colour = guide_legend(override.aes = list(size=6))) + 
        scale_x_continuous(label = function(x){format(x, scientific = TRUE)})
    
  }
  
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
    mutate(method = "Stan - Non-Centered") %>% 
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
    # bind_rows(get_stan_trace(stan_var, stan_list, sdev_conv)) %>% 
    bind_rows(get_all_python_traces(python_var, sdev_conv))
  
  return(df_combined_traces)
  
}

# fbm_var, stan_var, stan_list, python_var, py_sdev_conv

w1_traces <- join_all_traces("w1", "W\\[1", "desired_weight_vars", "w_ih") 
b1_traces <- join_all_traces("w2", "B\\[1", "desired_bias_vars", "b_h") 
w2_traces <- join_all_traces("w3", "W\\[2", "desired_weight_vars", "w_ho")
b2_traces <- join_all_traces("w4", "B\\[1,", "desired_bias_vars", "b_o")
hw1_traces <- join_all_traces("h1", "W_prec\\[1" , "weights_desired_hp_vars", "W_prec_ih", sdev_conv = T) %>% 
  filter(!(str_detect(name, "h1_2|h1_3|h1_4|h1_5")))
hb1_traces <- join_all_traces("h2", "B_prec\\[1", "biases_desired_hp_vars", "B_prec_h", sdev_conv = T) 
hw2_traces <- join_all_traces("h3", "W_prec\\[2", "weights_desired_hp_vars", "W_prec_ho", sdev_conv = T)
hb2_traces <- join_all_traces("h4", "B_prec\\[2", "biases_desired_hp_vars", "B_prec_o", sdev_conv = T)

# ARD traces

ard_traces <- join_all_traces("h1", "ard_prec\\[" , "ard_prec_vars", "ard_prec", sdev_conv = T)

ard_traces <- ard_traces %>% 
  filter(!(str_detect(name, "h1_1"))) %>% 
  mutate(name = ifelse(name == "ard_prec[1]" | name == "ard_prec_1", "h1_2", name)) %>% 
  mutate(name = ifelse(name == "ard_prec[2]" | name == "ard_prec_2", "h1_3", name)) %>% 
  mutate(name = ifelse(name == "ard_prec[3]" | name == "ard_prec_3", "h1_4", name)) %>% 
  mutate(name = ifelse(name == "ard_prec[4]" | name == "ard_prec_4", "h1_5", name)) %>% 
  left_join(tibble(new_name = c("X1", "X2", "X3", "X4"), 
                   name = c("h1_2", "h1_3", "h1_4", "h1_5")), 
            by = c("name")) %>% 
  select(-name) %>% 
  rename(name = new_name)

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

ard_trace_plot <- plot_traces(ard_traces, 
                              title = "Standard Deviation Hyperparameter: Input Automatic Relevance Determination", 
                              subtext = SUBTEXT, log = T, ard = T)

hb2_trace_plot <- plot_traces(hb2_traces, 
                              title = "Standard Deviation Hyperparameter: Output Unit Biases", subtext = SUBTEXT, log = T)

# y_prec_trace_plot <- plot_traces(y_prec_traces, 
#                                  title = "Standard Deviation Hyperparameter: Target Noise", 
#                                  subtext = SUBTEXT, 
#                                  size = 0.5, log = F)

# Append to one PDF ---------------------------------------------------------

all_plots <- list(
  # roc_plot,
  # y_prec_trace_plot,
  ard_trace_plot,
  hw1_trace_plot,
  hw2_trace_plot,
  hb1_trace_plot,
  hb2_trace_plot,
  w1_trace_plot,
  b1_trace_plot,
  w2_trace_plot,
  b2_trace_plot
)


OUTPUT_PATH <- "./output/multi_class/"
folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, type, folder_name)
dir.create(path)

ggsave(
  filename = str_c(path, "/", folder_name, "_", type, "_results.pdf"), 
  plot = marrangeGrob(all_plots, nrow=1, ncol=1), 
  width = 15, height = 9
)

print(path)