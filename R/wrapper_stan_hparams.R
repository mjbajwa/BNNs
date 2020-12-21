# Load Libraries ----------------------------------------------------------

library(rstan)
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyr)
library(gridExtra)

set.seed(10)

# User Inputs -------------------------------------------------------------
  
OUTPUT_PATH <- "./output/DEBUG/"
STAN_FILE <- "./stan/BNN_hparams_dev.stan"

# number of neurons per hidden layer.. e.g: c(8, 7) has two hidden layers with 8 and 7 hidden units respectively

G <- c(8) 
INFINITE_LIMIT <- c(0L)
HIERARCHICAL_FLAG_W <- 1
HIERARCHICAL_FLAG_B <- 0

# Prior definition using FBM language

FBM_W <- list("GAMMA_WIDTH" = rep(0.5, times = length(G) + 1),
              "GAMMA_ALPHA" = rep(2, times = length(G) + 1))

FBM_B <- list("GAMMA_WIDTH" = rep(0.5, times = length(G) + 1),
              "GAMMA_ALPHA" = rep(2, times = length(G) + 1))

FBM_Y <- list("GAMMA_WIDTH" = rep(0.5, times = 1),
              "GAMMA_ALPHA" = rep(5, times = 1))

# FBM to Stan Conversion of Prior -------------------------------------------------------

# TODO: add to utils.R

fbm_gamma_params_to_stan <- function(fbm_width, fbm_alpha){
  
  # TODO: check with Prof Neal this re-parametrization is correct.
  
  mean_precision = 1/(fbm_width^2)
  stan_alpha = fbm_alpha/2
  stan_beta = stan_alpha/mean_precision
  
  output = list("STAN_ALPHA" = stan_alpha,
                "STAN_BETA" = stan_beta)
  
  return(output)
  
}

# Convert FBM parameterization to STAN parametrization ---------------

# Shape = alpha, Scale = beta

W_STAN <- fbm_gamma_params_to_stan(FBM_W$GAMMA_WIDTH, FBM_W$GAMMA_ALPHA)
W_gamma_shape <- W_STAN$STAN_ALPHA
W_gamma_scale <- W_STAN$STAN_BETA

B_STAN <- fbm_gamma_params_to_stan(FBM_B$GAMMA_WIDTH, FBM_B$GAMMA_ALPHA)
B_gamma_shape <- B_STAN$STAN_ALPHA
B_gamma_scale <- B_STAN$STAN_BETA

Y_STAN <- fbm_gamma_params_to_stan(FBM_Y$GAMMA_WIDTH, FBM_Y$GAMMA_ALPHA)
Y_gamma_shape <- Y_STAN$STAN_ALPHA
Y_gamma_scale <- Y_STAN$STAN_BETA

# Check Prior

precision_w <- rgamma(n=1000, shape=W_gamma_shape[1], scale=1/W_gamma_scale[1])
precision_b <- rgamma(n=1000, shape=W_gamma_shape[1], scale=1/W_gamma_scale[1])
precision_y <- rgamma(n=1000, shape=W_gamma_shape[1], scale=1/W_gamma_scale[1])

hist(log10(1/sqrt(precision_w)), col="red2", main="Weights (log10 sdev)")
hist(log10(1/sqrt(precision_b)), col="red2", main="Biases (log10 sdev)")
hist(log10(1/sqrt(precision_y)), col="red2", main="Measurement Noise (log10 sdev)")

# Load Data -----------------------------------------------------------

df <- data.frame(read.table("./fbm_logs/rdata", header = FALSE))
colnames(df) <- c("X", "Y")
target_col <- "Y"
all_idx <- 1:nrow(df)
train_idx <- 1:100
test_idx <- all_idx[!(all_idx %in% train_idx)]

# Data pre-processing

X_train <- matrix(df[train_idx, names(df)[!str_detect(string = names(df), target_col)]])
y_train <- as.vector(matrix(df[train_idx, target_col]))
X_test <- matrix(df[test_idx, names(df)[!str_detect(string = names(df), target_col)]])
y_test <- as.vector(matrix(df[test_idx, target_col]))
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
  infinite_limit = array(INFINITE_LIMIT)
)

# Call Stan ---------------------------------------------------------------

fit <- stan(
  file = STAN_FILE,   
  data = data,  
  chains = 4, 
  warmup = 1000, 
  iter = 2000, 
  cores = 6, 
  refresh = 1,
  verbose = TRUE,
  seed = 1,
)

# Create dir to save outputs of the run

folder_name <- str_replace_all(Sys.time(), "-|:|\ ", "_")
path <- str_c(OUTPUT_PATH, folder_name)
dir.create(path)

# Utility Functions -------------------------------------------------------

# TODO: Port to utils.R

parse_stan_vars <- function(vars, stan_pattern="W", index_dim=3){
  
  #' Parses stan variables
  
  temp <- str_replace(vars, str_c(stan_pattern, "\\["), "") %>% 
    str_replace("\\]", "") %>% 
    str_split(",", n = index_dim) %>%
    lapply(function(x){c(x)})
  
  parsed_outputs <- matrix(data = 0, nrow=length(temp), ncol=index_dim)
  
  for(i in 1:length(temp)){
    parsed_outputs[i,] <- as.numeric(temp[i] %>% unlist())
  }
  
  df_parsed <- as_tibble(parsed_outputs)
  names(df_parsed) <- c("layer", "incoming_neuron", "outgoing_neuron")
  df_parsed <- df_parsed %>% 
    mutate(stan_var_name = vars) %>% 
    select(stan_var_name, everything())
  
  return(df_parsed) 
  
}

markov_chain_samples <- function(stan_fit, var, n_chains=4, burn_in=1000, iters=2000){
  
  #' Example input values
  
  # n_chains = 4
  # burn_in = 1000
  # iters = 2000
  # var = "W[1,1,1]"
  
  # Create empty dataframe to save results
  
  df_chains <- as_tibble(matrix(data=0, nrow = 2000, ncol=4))
  names(df_chains) <- as.vector(unlist(lapply(as.list(1:n_chains), function(x){paste("chain_", x, sep="")})))
  
  # Loop through each chain and extract the samples
  
  for(chain in 1:n_chains){
    
    temp <- stan_fit@sim$samples[[chain]]
    parameter_samples <- temp[[var]]
    
    df_chains[as.character(paste("chain_", chain, sep=""))] <- parameter_samples
    
  }
  
  # Calculate across chain averages (TODO: might also want to do within chain averages for R_hat)
  
  df_chains["time_index"] = 1:nrow(df_chains)
  
  df_chain_summary <- df_chains %>% 
    tidyr::pivot_longer(!time_index) %>% 
    group_by(time_index) %>%
    summarize(mean_chains = mean(value),
              sdev_chains = sd(value))
  
  # Join the summary with final
  
  df_chains["var"] = var
  df_chains["stationary"] = ifelse(1:nrow(df_chains) > burn_in, T, F)
  
  df_chains <- df_chains %>% 
    left_join(df_chain_summary, by = "time_index")
  
  return(df_chains)
  
}

mcmc_trace_plot <- function(df_mcmc_param, var, burn_in=1000){
  
  df_plot <- df_mcmc_param %>% 
    select(time_index, contains("chain_")) %>% 
    tidyr::pivot_longer(!time_index, names_to = "chain") %>% 
    mutate(stationary = ifelse(time_index > 1000, T, F))
  
  ggplot(df_plot) +
    geom_point(aes(x = time_index, y = value, color = chain, alpha = stationary), size=0.5) + 
    scale_alpha_manual(values=c(0.05, 1)) + 
    geom_vline(xintercept = burn_in,
               linetype=2) + 
    theme_bw() + 
    ggtitle(str_c(unique(df_mcmc_param$var))) + 
    xlab("") + 
    ylab("") + 
    theme(text=element_text(size=16),
          legend.position = "none")
  
}

# Postprocessing -------------------------------------------------------------

# Extract summary table from the fit object

stan_summary <- summary(fit)
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

W_sdev_names <- all_parameters[stringr::str_detect(all_parameters, "W_sdev")]
W_precision_names <- all_parameters[stringr::str_detect(all_parameters, "W_prec")]
B_sdev_names <- all_parameters[stringr::str_detect(all_parameters, "B_sdev")]
B_precision_names <- all_parameters[stringr::str_detect(all_parameters, "B_prec")]

# predicted values of test set

y_train_names <- all_parameters[str_detect(all_parameters, "y_train_pred")]
y_test_names <- all_parameters[str_detect(all_parameters, "y_test_pred")]

# Prediction of training and test set ----------------------------------------------------

df_x_train <- as_tibble(X_train) %>% 
  rename_all(function(x) paste0("X_", x))

df_x_test <- as_tibble(X_test) %>% 
  rename_all(function(x) paste0("X_", x))

#' Assess predictive distributions - is poor sampling having an effect? 

df_post_train <- df_stan_summary %>% 
  filter(str_detect(stan_var_name, "y_train_pred")) %>% 
  mutate(actual = y_train) %>% 
  select(stan_var_name, mean, actual, everything()) %>% 
  mutate(label = "train") %>% 
  bind_cols(df_x_train)

df_post_test <- df_stan_summary %>% 
  filter(str_detect(stan_var_name, "y_test_pred")) %>% 
  mutate(actual = y_test) %>% 
  select(stan_var_name, mean, actual, everything()) %>% 
  mutate(label = "test") %>% 
  bind_cols(df_x_test)

df_post_preds <- df_post_train %>% 
  bind_rows(df_post_test)

pred_actual_plot <- ggplot(df_post_preds) + 
  geom_point(aes(x = actual, y = mean, color=label), alpha = 0.5, size = 2) + 
  geom_linerange(aes(x = actual, ymin =`2.5%`, ymax= `97.5%`), alpha = 0.5) + 
  scale_color_manual(values = c("red2", "green3")) + 
  geom_abline(slope=1,intercept=0) + 
  theme_bw() + 
  xlab("Actual") + 
  ylab("Predicted") + 
  theme(text = element_text(size=16)) + 
  facet_wrap(label~., scales = "free_x") + 
  ggtitle("Predicted vs. Actual")

yx_filtered_plot <- ggplot(df_post_preds %>% filter(X_V1 > -2.2)) + 
  geom_ribbon(aes(x = X_V1, ymin =`2.5%`, ymax= `97.5%`, fill=label), alpha = 0.3) + 
  geom_point(aes(x = X_V1, y = mean, color=label), alpha = 0.5, size = 2) + 
  geom_point(aes(x = X_V1, y = actual), color = "black", alpha = 0.2, size = 2) + 
  scale_color_manual(values = c("red2", "green3")) + 
  theme_bw() + 
  xlab("X") + 
  ylab("Y (Predicted)") + 
  theme(text = element_text(size=16)) + 
  ggtitle("Y vs. X", subtitle = "Filtered X < -2.2")

yx_unfiltered_plot <- ggplot(df_post_preds) + #%>% filter(X_V1 > -2.2)) + 
  geom_ribbon(aes(x = X_V1, ymin =`2.5%`, ymax= `97.5%`, fill=label), alpha = 0.3) + 
  geom_point(aes(x = X_V1, y = actual, color=label), alpha = 0.5, size = 2) + 
  scale_color_manual(values = c("red2", "green4")) + 
  scale_fill_manual(values = c("red2", "green4")) + 
  theme_bw() + 
  xlab("X") + 
  ylab("Y (Predicted)") + 
  theme(text = element_text(size=16)) + 
  facet_wrap(label~., scales = "free") + 
  ggtitle("Y vs. X")

# ggplot(df_post_preds) +
#   geom_ribbon(aes(x = X_V1, ymin =`2.5%`, ymax= `97.5%`, fill=label), alpha = 0.3) +
#   geom_point(aes(x = X_V1, y = mean, color=label), alpha = 0.5, size = 2) +
#   geom_point(aes(x = X_V1, y = actual), color="black", alpha = 0.1, size = 1) +
#   scale_color_manual(values = c("red2", "green3")) +
#   theme_bw() +
#   xlab("X") +
#   ylab("Y (Predicted)") +
#   theme(text = element_text(size=16)) +
#   facet_wrap(label~., scales = "free") +
#   ggtitle("Y vs. X", subtitle = "No filtered points")

df_post_preds$Rhat %>% hist(col = "red2")

# Extremely Paradoxical.... Why do the predictions feel good but the actual parameters end up diverging?

# Trace Plot Analysis of MCMC for weights/biases ---------------------------------------------

# Parsing the distribution of weights in a clean way ---

weights_parsed <- parse_stan_vars(hidden_weights_names, "W", 3)
biases_parsed <- parse_stan_vars(hidden_biases_names, "B", 2)
#W_sdev_parsed <- parse_stan_vars(W_sdev_names, "W_sdev", 3)
#W_precision_parsed <- parse_stan_vars(W_precision_names, "W_precision", 3)
#B_sdev_parsed <- parse_stan_vars(B_sdev_names, "B_sdev", 2)
#B_precision_parsed <- parse_stan_vars(B_precision_names, "B_precision", 2)

# Weights analysis

df_weights_posterior <- weights_parsed %>% 
  left_join(df_stan_summary, by = "stan_var_name")

df_biases_posterior <- biases_parsed %>% 
  left_join(df_stan_summary, by = "stan_var_name")

# Parse out important weight parameters 
# TODO: Do this in a more generalizable way

layer_1_weights <- df_weights_posterior %>% 
  filter(layer == 1,
         incoming_neuron == 1) %>% 
  pull(stan_var_name)

layer_2_weights <- df_weights_posterior %>% 
  filter(layer == 2,
         outgoing_neuron == 1) %>% 
  pull(stan_var_name)

desired_weight_vars <- c(layer_1_weights, layer_2_weights)
weight_trace_plots = list()

for(i in 1:length(desired_weight_vars)){
  var <- desired_weight_vars[i]
  weight_trace_plots[[i]] <- markov_chain_samples(fit, var) %>% 
    mcmc_trace_plot(var, burn_in = 1000)
}

# Save Results ------------------------------------------------------------

# Prediction Plots

ggsave(str_c(path, "/predicted_vs_actual.png"),
       pred_actual_plot,
       width = 11,
       height = 8)

ggsave(str_c(path, "/y_vs_x_filtered.png"),
       yx_filtered_plot,
       width = 11,
       height = 8)

ggsave(str_c(path, "/y_vs_x_unfiltered.png"),
       yx_unfiltered_plot,
       width = 11,
       height = 8)

# Weight Plots

png(str_c(path, "/weights_ih.png"), width = 14, height = 8, units = "in", res=50)
do.call("grid.arrange", c(weight_trace_plots[1:8], 
                          ncol = floor(sqrt(length(weight_trace_plots)))))
dev.off()

png(str_c(path, "/weights_ho.png"), width = 14, height = 8, units = "in", res=50)
do.call("grid.arrange", c(weight_trace_plots[9:16], 
        ncol = floor(sqrt(length(weight_trace_plots)))))
dev.off()
