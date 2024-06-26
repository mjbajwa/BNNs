# Multi-class Binary classification problem

# Load Libraries ----------------------------------------------------------

library(yaml)
library(rstan)
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyr)
library(gridExtra)
library(reshape2)
library(readr)
library(ROCit)
set.seed(42)

source("R/stan_utils.R")

# Load YAML ---------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
YAML_INPUTS <- read_yaml("config/multi_class/centered.yaml") # yaml::read_yaml(args[1])

# Paths -------------------------------------------------------------------

OUTPUT_PATH <- YAML_INPUTS$OUTPUT_PATH
STAN_FILE <- YAML_INPUTS$STAN_FILE
INPUT_TYPE <- YAML_INPUTS$INPUT_TYPE
SCALE_INPUT <- YAML_INPUTS$SCALE_INPUT
TRAIN_FRACTION <- YAML_INPUTS$TRAIN_FRACTION

# User Inputs -------------------------------------------------------------

# Architecture Design

G <- YAML_INPUTS$G
INFINITE_LIMIT <- YAML_INPUTS$INFINITE_LIMIT
HIERARCHICAL_FLAG_W <- YAML_INPUTS$HIERARCHICAL_FLAG_W
HIERARCHICAL_FLAG_B <- YAML_INPUTS$HIERARCHICAL_FLAG_B
FIX_TARGET_NOISE <- YAML_INPUTS$FIX_TARGET_NOISE
SAMPLE_FROM_PRIOR <- YAML_INPUTS$SAMPLE_FROM_PRIOR

INIT_FUN <- function(...) {
  if (str_detect(YAML_INPUTS$STAN_FILE, "log")) {
    list(
      log_W_prec = log(YAML_INPUTS$INIT$WEIGHTS),
      log_B_prec = log(YAML_INPUTS$INIT$BIASES),
      log_y_prec = log(YAML_INPUTS$INIT$TARGET_NOISE),
      W = array(0, dim = c(
        length(G) + 1, max(max(G), 1), max(max(G), 1)
      )),
      B = array(0, dim = c(max(max(
        G
      ), 1), length(G) + 1)),
      W_raw = array(0, dim = c(
        length(G) + 1, max(max(G), 1), max(max(G), 1)
      )),
      B_raw = array(0, dim = c(max(max(
        G
      ), 1), length(G) + 1))
    )
    
  } else {
    
    list(
      W_prec = YAML_INPUTS$INIT$WEIGHTS,
      B_prec = as.array(YAML_INPUTS$INIT$BIASES),
      y_prec = YAML_INPUTS$INIT$TARGET_NOISE,
      ard_prec = as.array(YAML_INPUTS$INIT$ARD),
      ard_prec_raw = as.array(YAML_INPUTS$INIT$ARD),
      W = array(0, dim = c(
        length(G) + 1, max(max(G), 1), max(max(G), 1)
      )),
      B = array(0, dim = c(max(max(
        G
      ), 1), length(G) + 1)),
      W_raw = array(0, dim = c(
        length(G) + 1, max(max(G), 1), max(max(G), 1)
      )),
      B_raw = array(0, dim = c(max(max(
        G
      ), 1), length(G) + 1))
    )
    
    
  }
  
}

# Prior definition using FBM language

FBM_W <- list(
  "GAMMA_WIDTH" = YAML_INPUTS$PRIORS$WEIGHTS$WIDTH,
  "GAMMA_ALPHA" = YAML_INPUTS$PRIORS$WEIGHTS$ALPHA
)

FBM_B <- list(
  "GAMMA_WIDTH" = YAML_INPUTS$PRIORS$BIASES$WIDTH,
  "GAMMA_ALPHA" = YAML_INPUTS$PRIORS$BIASES$ALPHA
)

FBM_Y <- list(
  "GAMMA_WIDTH" = YAML_INPUTS$PRIORS$TARGET$WIDTH,
  "GAMMA_ALPHA" = YAML_INPUTS$PRIORS$TARGET$ALPHA
)

# MCMC control for stan

MCMC_INPUTS <- YAML_INPUTS$MCMC_INPUTS

# Constants ---------------------------------------------------------------

folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, "stan_", folder_name)
dir.create(path)

INPUTS <- list(
  "STAN_FILE" = STAN_FILE,
  "G" = G,
  "INFINITE_LIMIT" = INFINITE_LIMIT,
  "HIERARCHICAL_FLAG_W" = HIERARCHICAL_FLAG_W,
  "HIERARCHICAL_FLAG_B" = HIERARCHICAL_FLAG_B,
  "FBM_W" = FBM_W,
  "FBM_B" = FBM_B,
  "FBM_Y" = FBM_Y,
  "INPUT_TYPE" = INPUT_TYPE,
  "SCALE_INPUT" = SCALE_INPUT,
  "TRAIN_FRACTION" = TRAIN_FRACTION,
  "INIT_VALUES" = INIT_FUN(),
  "FIX_TARGET_NOISE" = FIX_TARGET_NOISE,
  "SAMPLE_FROM_PRIOR" = SAMPLE_FROM_PRIOR
)

# capture.output(c(INPUTS, MCMC_INPUTS), file = str_c(path, '/inputs.txt'))

# Convert FBM parameterization to STAN parametrization ---------------

# Shape = alpha, Scale = beta

W_STAN <- fbm_gamma_params_to_stan(FBM_W$GAMMA_WIDTH, FBM_W$GAMMA_ALPHA)
W_gamma_shape <- W_STAN$STAN_ALPHA
W_gamma_scale <- W_STAN$STAN_BETA

# TODO: Make this more elegant. Updating the priors to adjust for infinite limit. 
W_gamma_scale[[1]] <- fbm_gamma_params_to_stan(FBM_W$GAMMA_WIDTH[[1]], FBM_W$GAMMA_ALPHA[[1]], 4^2)$STAN_BETA
W_gamma_scale[[2]] <- fbm_gamma_params_to_stan(FBM_W$GAMMA_WIDTH[[2]], FBM_W$GAMMA_ALPHA[[2]], 8)$STAN_BETA

B_STAN <- fbm_gamma_params_to_stan(FBM_B$GAMMA_WIDTH, FBM_B$GAMMA_ALPHA)
B_gamma_shape <- B_STAN$STAN_ALPHA
B_gamma_scale <- B_STAN$STAN_BETA

Y_STAN <- fbm_gamma_params_to_stan(FBM_Y$GAMMA_WIDTH, FBM_Y$GAMMA_ALPHA)
Y_gamma_shape <- Y_STAN$STAN_ALPHA
Y_gamma_scale <- Y_STAN$STAN_BETA

# ARD_STAN <- fbm_gamma_params_to_stan(FBM_ARD$GAMMA_WIDTH, FBM_ARD$GAMMA_ALPHA)
# ARD_gamma_shape <- ARD_STAN$STAN_ALPHA
# ARD_gamma_scale <- ARD_STAN$STAN_BETA
# ARD_gamma_scale <- rep(fbm_gamma_params_to_stan(FBM_ARD$GAMMA_WIDTH[[2]], FBM_ARD$GAMMA_ALPHA[[2]], 8)$STAN_BETA, 
#                        length(ARD_gamma_scale))

# Check Prior

# precision_w <- rgamma(n = 1000, shape = W_gamma_shape[1], scale = 1 / W_gamma_scale[1])
# precision_b <- rgamma(n = 1000, shape = B_gamma_shape[1], scale = 1 / B_gamma_scale[1])
# precision_y <- rgamma(n = 1000, shape = Y_gamma_shape[1], scale = 1 / Y_gamma_scale[1])
#
# # par(mfrow = c(3, 1))
# # hist(log10(1 / sqrt(precision_w)), col = "red2", main = "Weights (log10 sdev)")
# # hist(log10(1 / sqrt(precision_b)), col = "red2", main = "Biases (log10 sdev)")
# # hist(log10(1 / sqrt(precision_y)), col = "red2", main = "Measurement Noise (log10 sdev)")

# Load Data -----------------------------------------------------------

df <- data.frame(read.table("./data/cdata", header = FALSE))
colnames(df) <- c("V1", "V2", "V3", "V4", "Y")
df <- as_tibble(df)
df$Y <- as.integer(df$Y + 1)

all_idx <- 1:nrow(df)
train_idx <- 1:floor(TRAIN_FRACTION * max(all_idx))
test_idx <- all_idx[!(all_idx %in% train_idx)]
target_col <- "Y"

# Data pre-processing

X_train <- df %>% as_tibble() %>% slice(train_idx) %>% select(-contains("Y"))
y_train <- df %>% as_tibble() %>% slice(train_idx) %>% select(contains("Y")) %>% pull()
X_test <- df %>% as_tibble() %>% slice(-train_idx) %>% select(-contains("Y"))
y_test <- df %>% as_tibble() %>% slice(-train_idx) %>% select(contains("Y")) %>% pull()
N <- nrow(X_train) # number of observations in training data
K <- ncol(X_train) # number of input features
N_test <- nrow(X_test) # number of observations in test data

data = list(
  N = N,
  K = K,
  h = length(G),
  G = array(G),
  X_train = X_train,
  y_train = y_train,
  N_test = N_test,
  X_test = X_test,
  W_gamma_shape = W_gamma_shape,
  W_gamma_scale = W_gamma_scale,
  B_gamma_shape = B_gamma_shape,
  B_gamma_scale = B_gamma_scale,
  y_gamma_shape = Y_gamma_shape,
  y_gamma_scale = Y_gamma_scale,
  use_hierarchical_w = HIERARCHICAL_FLAG_W,
  use_hierarchical_b = HIERARCHICAL_FLAG_B,
  infinite_limit = array(INFINITE_LIMIT),
  fix_target_noise = FIX_TARGET_NOISE,
  sample_from_prior = SAMPLE_FROM_PRIOR
  # ard_shape = ARD_gamma_shape, 
  # ard_scale = ARD_gamma_scale
)

# Call Stan ---------------------------------------------------------------

fit <- stan(
  file = STAN_FILE,
  data = data,
  init = INIT_FUN,
  chains = MCMC_INPUTS$CHAINS,
  warmup = MCMC_INPUTS$BURN_IN,
  iter = MCMC_INPUTS$ITER,
  cores = MCMC_INPUTS$CORES,
  verbose = FALSE,
  refresh = 10,
  seed = 42,
  # set to 42 for all previous results (date < Feb-04).
  control = MCMC_INPUTS$CONTROL
)

# Postprocessing -------------------------------------------------------------

# Extract summary table from the fit object

stan_summary <- summary(fit, probs = c(0.01, 0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975, 0.99))
all_parameters <- attr(stan_summary$summary, "dimnames")[[1]]
df_stan_summary <- stan_summary$summary %>%
  as_tibble() %>%
  mutate(stan_var_name = all_parameters)

# Weight Parameter Names

hidden_weights_names <- all_parameters[stringr::str_detect(all_parameters, "W") &
                 !stringr::str_detect(all_parameters, "W_gamma|W_sdev|W_prec")]

# Biases Parameter Names

hidden_biases_names <- all_parameters[stringr::str_detect(all_parameters, "B") &
                 !stringr::str_detect(all_parameters, "B_gamma|B_sdev|B_prec")]

# Weight Standard Deviation (names)

W_precision_names <- all_parameters[stringr::str_detect(all_parameters, "W_prec")]
B_precision_names <- all_parameters[stringr::str_detect(all_parameters, "B_prec")]
ard_precision_names <- all_parameters[stringr::str_detect(all_parameters, "ard_prec")]

# predicted values of test set

y_train_names <- all_parameters[str_detect(all_parameters, "y_train_pred_final")]
y_test_names <- all_parameters[str_detect(all_parameters, "y_test_pred")]
y_precision_names <- all_parameters[str_detect(all_parameters, "y_prec")]

df_stan_summary %>% filter(str_detect(stan_var_name, "y_train_prob")) # %>% pull(mean) %>% plot(col = "red2")

# Prediction of training and test set ----------------------------------------------------

df_x_train <- as_tibble(X_train) %>%
  rename_all(function(x)
    paste0("X_", x))

df_x_test <- as_tibble(X_test) %>%
  rename_all(function(x)
    paste0("X_", x))

#' Assess predictive distributions - is poor sampling having an effect?

df_post_train <- df_stan_summary %>%
  filter(str_detect(stan_var_name, "y_train_pred")) %>%
  mutate(actual = y_train) %>%
  select(stan_var_name, mean, actual, everything()) %>%
  mutate(label = "train") %>%
  bind_cols(df_x_train) %>% 
  rename(predicted = `50%`)

df_post_test <- df_stan_summary %>%
  filter(str_detect(stan_var_name, "y_test_pred")) %>%
  mutate(actual = y_test) %>%
  select(stan_var_name, mean, actual, everything()) %>%
  mutate(label = "test") %>%
  bind_cols(df_x_test) %>% 
  rename(predicted = `50%`)

error <- sum(pmin(abs(df_post_test$predicted - df_post_test$actual), 1))/nrow(df_post_test)*100
print(str_c("Error: ", error, "%"))

threshold <- 0.60

df_post_preds <- df_post_train %>%
  bind_rows(df_post_test) %>% 
  mutate(mean_prob = 1/(1 + exp(-mean)))

tmp <- df_post_preds %>%
  select(X_V1, X_V2, actual, mean_prob, sd, label) %>% 
  mutate(predicted = ifelse(mean_prob >= threshold, 1L, 0L)) %>%  
  tidyr::pivot_longer(cols = c("actual", "predicted"))

pred_actual_plot <- ggplot(tmp %>% filter(label == "test")) + 
  geom_point(aes(x = X_V1, y = X_V2, color = as.character(value)), size = 2) + 
  theme_bw() + 
  scale_color_manual(values = c("red3", "green3")) + 
  facet_wrap(name~.)

roc_train_stan <- rocit(score = df_post_preds %>% filter(label == "train") %>% pull(mean_prob), 
                        class = df_post_preds %>% filter(label == "train") %>% pull(actual))

roc_test_stan <- rocit(score = df_post_preds %>% filter(label == "test") %>% pull(mean_prob), 
                       class = df_post_preds %>% filter(label == "test") %>% pull(actual))

ggplot() + 
  geom_line(aes(x = roc_test_stan$FPR, y = roc_test_stan$TPR)) + 
  xlab("False Positive Rate - FPR") + 
  ylab("True Positive Rate - TPR") + 
  ggtitle("ROC curve for Test set (Stan)", subtitle = str_c("AUC: ", roc_test_stan$AUC)) + 
  theme_bw()
# 
# tmp2 <- df_post_preds %>%
#   select(X_V1, X_V2, actual, mean_prob, sd, label) %>% 
#   mutate(predicted = ifelse(mean_prob >= threshold, 1L, 0L))
# 
# a = tmp2 %>% filter(label == "test") %>% pull(predicted)
# b = tmp2 %>% filter(label == "test") %>% pull(actual)

# Trace Plot Analysis of MCMC for weights/biases ---------------------------------------------

weights_parsed <- parse_stan_vars(hidden_weights_names, "W", 3)
W_precision_parsed <-
  parse_stan_vars(W_precision_names, "W_prec", 1, column_names = c("layer"))
biases_parsed <-
  parse_stan_vars(hidden_biases_names, "B", 2, column_names = c("neuron", "layer"))
B_precision_parsed <-
  parse_stan_vars(B_precision_names, "B_prec", 1, column_names = c("layer"))
ard_precision_parsed <- parse_stan_vars(ard_precision_names, "ard_prec", 1, column_names = c("layer")) %>% 
  mutate(layer = "1")

# Weights analysis

df_weights_posterior <- weights_parsed %>%
  left_join(df_stan_summary, by = "stan_var_name")

df_biases_posterior <- biases_parsed %>%
  left_join(df_stan_summary, by = "stan_var_name")

df_weights_hp_posterior <- W_precision_parsed %>%
  left_join(df_stan_summary, by = "stan_var_name")

df_biases_hp_posterior <- B_precision_parsed %>%
  left_join(df_stan_summary, by = "stan_var_name")

df_noise_hp_posterior <- df_stan_summary %>%
  filter(str_detect(stan_var_name, "y_prec"))

df_ard_posterior <- df_stan_summary %>%
  filter(str_detect(stan_var_name, "ard_prec"))

# Parse out important weight and bias parameters (that are not redundant)

desired_weight_vars <- c()

for (l in 1:(length(G) + 1)) {
  # Define vector of incoming hidden unit by layer
  
  if (l == 1) {
    previous_hidden_units = 1:ncol(X_train)
  } else {
    previous_hidden_units = 1:G[l - 1]
  }
  
  # Define vector of outgoing hidden unit by layer
  
  if (l == length(G) + 1) {
    next_hidden_units = 1:3
  } else {
    next_hidden_units = 1:G[l]
  }
  
  # Define weights of layers
  
  layer_weights <- df_weights_posterior %>%
    filter(
      layer == l,
      incoming_neuron %in% previous_hidden_units,
      outgoing_neuron %in% next_hidden_units
    ) %>%
    filter(!str_detect(stan_var_name, "raw")) %>%
    pull(stan_var_name)
  
  desired_weight_vars <- c(desired_weight_vars, layer_weights)
  
}

desired_bias_vars <- c()

for (l in 1:(length(G) + 1)) {
  if (l == length(G) + 1) {
    next_hidden_units = 1:3
  } else {
    next_hidden_units = 1:G[l]
  }
  
  # Define weights of layers
  
  layer_weights <- df_biases_posterior %>%
    filter(layer == l,
           neuron %in% next_hidden_units,) %>%
    filter(!str_detect(stan_var_name, "raw")) %>%
    pull(stan_var_name)
  
  desired_bias_vars <- c(desired_bias_vars, layer_weights)
  
}

##  Trace Plots

# Weights

weight_trace_plots <- list()

for (i in 1:length(desired_weight_vars)) {
  var <- desired_weight_vars[i]
  
  weight_trace_plots[[var]] <- markov_chain_samples(fit,
                                                    var,
                                                    burn_in = MCMC_INPUTS$BURN_IN,
                                                    iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN)
  
}


# Weight hyperparameters

weights_desired_hp_vars <- df_weights_hp_posterior$stan_var_name
weights_hp_trace_plots <- list()

for (i in 1:length(weights_desired_hp_vars)) {
  var <- weights_desired_hp_vars[i]
  
  weights_hp_trace_plots[[var]] <- markov_chain_samples(fit,
                                                        var,
                                                        burn_in = MCMC_INPUTS$BURN_IN,
                                                        iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN, min_time = 1)
  
}

# Biases

bias_trace_plots <- list()

for (i in 1:length(desired_bias_vars)) {
  var <- desired_bias_vars[i]
  
  bias_trace_plots[[var]] <- markov_chain_samples(fit,
                                                  var,
                                                  burn_in = MCMC_INPUTS$BURN_IN,
                                                  iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN)
  
}

# Bias Hyperparameters

biases_desired_hp_vars <- df_biases_hp_posterior$stan_var_name
biases_hp_trace_plots <- list()

for (i in 1:length(biases_desired_hp_vars)) {
  var <- biases_desired_hp_vars[i]
  
  biases_hp_trace_plots[[var]] <- markov_chain_samples(fit,
                                                       var,
                                                       burn_in = MCMC_INPUTS$BURN_IN,
                                                       iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN, min_time = 1)
  
}

# Target noise

target_noise_hp_vars <- df_noise_hp_posterior$stan_var_name
target_noise_trace_plots <- list()

for (i in 1:length(target_noise_hp_vars)) {
  var <- target_noise_hp_vars[i]
  
  target_noise_trace_plots[[var]] <- markov_chain_samples(fit,
                                                          var,
                                                          burn_in = MCMC_INPUTS$BURN_IN,
                                                          iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN, min_time = 1)
  
}

# ARD precision 

ard_prec_vars <- df_ard_posterior$stan_var_name
ard_prec_trace_plots <- list()

for (i in 1:length(ard_prec_vars)) {
  var <- ard_prec_vars[i]
  
  ard_prec_trace_plots[[var]] <- markov_chain_samples(fit,
                                                          var,
                                                          burn_in = MCMC_INPUTS$BURN_IN,
                                                          iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN, min_time = 1)
  
}

# Target noise

target_noise_hp_vars <- df_noise_hp_posterior$stan_var_name
target_noise_trace_plots <- list()

for (i in 1:length(target_noise_hp_vars)) {
  var <- target_noise_hp_vars[i]
  
  target_noise_trace_plots[[var]] <- markov_chain_samples(fit,
                                                          var,
                                                          burn_in = MCMC_INPUTS$BURN_IN,
                                                          iters = MCMC_INPUTS$ITER) %>%
    mcmc_trace_plot(var, burn_in = MCMC_INPUTS$BURN_IN, min_time = 1)
  
}

# Get Rejection Rate ------------------------------------------------------

sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
df_samples <- tibble()

for (chain in 1:length(sampler_params)) {
  df_temp <-
    sampler_params[[chain]] %>% as_tibble() %>% mutate(chain = chain)
  df_samples <- df_samples %>% bind_rows(df_temp)
  
}

df_plot_stats_mean <- df_samples %>%
  group_by(chain) %>%
  summarise_all(.funs = list("mean" = mean))

df_plot_stats <- df_samples %>%
  pivot_longer(
    cols = c(
      "accept_stat__",
      "stepsize__",
      "treedepth__",
      "n_leapfrog__",
      "divergent__",
      "energy__"
    )
  )

chain_statistics <- df_plot_stats %>%
  mutate(chain = as.character(chain),
         name = str_replace(name, "__", "")) %>%
  ggplot() +
  geom_boxplot(aes(x = chain, y = value, fill = name), alpha = 0.5) +
  facet_wrap(name ~ ., scales = "free") +
  theme_bw() +
  ylab("") +
  ggtitle("Markov chain Monte Carlo - statistics of key attributes for NUTS") +
  theme(text = element_text(size = 18),
        legend.position = "none")

capture.output(df_plot_stats_mean, file = str_c(path, '/chain_stats.txt'))

# Step-Size Behavior During Sampling --------------------------------

sampler_params <- get_sampler_params(fit, inc_warmup = TRUE)
df_samples <- tibble()

for (chain in 1:length(sampler_params)) {
  df_temp <-
    sampler_params[[chain]] %>% as_tibble() %>% mutate(chain = as.character(chain), iter = 1:n())
  df_samples <- df_samples %>% bind_rows(df_temp)
  
}

stepsize_plots <-
  ggplot(df_samples %>% filter(iter %in% 100:MCMC_INPUTS$ITER)) +
  geom_line(aes(x = iter, y = stepsize__, color = chain),
            alpha = 0.8) +
  theme_bw() +
  facet_wrap(chain ~ ., scales = "free") +
  xlab("Iteration") +
  ylab("Step Size")

# Save Results ------------------------------------------------------------

if (YAML_INPUTS$SAVE_PLOTS) {
  # Prediction Plots
  
  ggsave(
    str_c(path, "/predicted_vs_actual.png"),
    pred_actual_plot,
    width = 11,
    height = 8
  )
  
  # Weights Traces
  
  png(
    str_c(path, "/", "weight_traces", ".png"),
    width = 20,
    height = 12,
    units = "in",
    res = 100
  )
  do.call("grid.arrange", weight_trace_plots)
  dev.off()
  
  # Weights hyperparameters
  
  png(
    str_c(path, "/", "weights_hp_traces", ".png"),
    width = 20,
    height = 12,
    units = "in",
    res = 100
  )
  do.call("grid.arrange", weights_hp_trace_plots)
  dev.off()
  
  
  # Biases
  
  png(
    str_c(path, "/", "biases_traces", ".png"),
    width = 20,
    height = 12,
    units = "in",
    res = 100
  )
  do.call("grid.arrange", bias_trace_plots)
  dev.off()
  
  # Biases hyperparameters
  
  png(
    str_c(path, "/", "biases_hp_traces", ".png"),
    width = 20,
    height = 12,
    units = "in",
    res = 100
  )
  do.call("grid.arrange", biases_hp_trace_plots)
  dev.off()
  
  # Target noise
  
  png(
    str_c(path, "/", "target_noise_trace", ".png"),
    width = 20,
    height = 12,
    units = "in",
    res = 100
  )
  do.call("grid.arrange", target_noise_trace_plots)
  dev.off()
  
  # Stepsize and other auxillaries
  
  ggsave(
    str_c(path, "/stepsize_plots.png"),
    stepsize_plots,
    width = 11,
    height = 8
  )
  
  ggsave(
    str_c(path, "/chain_statistics.png"),
    chain_statistics,
    width = 11,
    height = 8
  )
  
}

# Categorical Probabilities -----------------------------------------------

multi_class_extract_prob <- function(stan_summary, all_parameters, pattern = "y_train_prob"){
  
  tmp_tbl <- stan_summary$c_summary[ , , 1] %>% 
    as_tibble() %>% 
    mutate(stan_var_name = all_parameters) %>% 
    filter(str_detect(stan_var_name, pattern)) 
  
  tmp_vars <- tmp_tbl %>% 
    pull(stan_var_name)
  
  y_train_prob <- parse_stan_vars(vars = tmp_vars, stan_pattern = pattern, index_dim = 2, 
                                  column_names = c("index", "category"))
  
  tmp_tbl %>% 
    left_join(y_train_prob, by = "stan_var_name") %>% 
    mutate(stan_var_name = pattern) %>% 
    select(index, category, mean) %>% 
    tidyr::pivot_wider(id_cols = index, names_from = category, values_from = mean, names_prefix = "mean_prob_")
  
}

df_prob_train <-  multi_class_extract_prob(stan_summary, all_parameters, pattern = "y_train_prob") %>% mutate(label = "train")
df_prob_test <- multi_class_extract_prob(stan_summary, all_parameters, pattern = "y_test_prob") %>% mutate(label = "test")
df_prob <- bind_rows(df_prob_train, df_prob_test)

# Return Object -----------------------------------------------------------

outputs <- list(
  "stan_file" = STAN_FILE,
  "inputs" = INPUTS,
  "outputs" = list(
    "stan_fit" = fit,
    "df_predictions" = df_post_preds,
    "df_prob" = df_prob,
    "desired_weight_vars" = desired_weight_vars,
    "desired_bias_vars" = desired_bias_vars,
    "weights_desired_hp_vars" = weights_desired_hp_vars,
    "biases_desired_hp_vars" = biases_desired_hp_vars,
    "target_noise_hp_vars" = target_noise_hp_vars,
    "ard_prec_vars" = ard_prec_vars, 
    "df_chain_statistics" = df_samples
  )
)

write_rds(outputs, str_c(path, "/outputs.rds"))
print(str_c("Stan Results: ", path))
# rm(list = ls())