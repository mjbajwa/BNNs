library(rstan)
library(dplyr)
library(stringr)
library(ggplot2)

# Architecture Design

G <- c(8) # number of neurons

# Generate Data -----------------------------------------------------------

df <- data.frame(read.table("./fbm_logs/rdata", header = FALSE))
colnames(df) <- c("X", "Y")
target_col <- "Y"
all_idx <- 1:nrow(df)
train_idx <- 1:100
test_idx <- all_idx[!(all_idx %in% train_idx)]

# Data preprocessing

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
  X_test = X_test
)

# Call Stan ---------------------------------------------------------------

fit <- stan(
  file = "stan/BNN.stan",  # Stan program
  data = data,    # named list of data
  chains = 4, 
  warmup = 1000, 
  iter = 2000, 
  cores = 6, 
  refresh = 1,
  verbose = TRUE
)

# Fit Diagnostics ---------------------------------------------------------

# summary(fit)
# rstan::plot(fit)
# rstan::stan_trace(fit)
# rstan::stan_plot(fit)
rstan::stan_rhat(fit)
# rstan::stan_dens(fit, separate_chains = T)

# Postprocessing -------------------------------------------------------------

# Extract certain properties from the fit object

stan_summary <- summary(fit)
all_parameters <- attr(stan_summary$summary, "dimnames")[[1]]

# Weight Parameter Names

hidden_weights <- all_parameters[stringr::str_detect(all_parameters, "W") & 
                 !stringr::str_detect(all_parameters, "W_initial|W_final")]

# Biases Parameter Names

hidden_biases <- all_parameters[stringr::str_detect(all_parameters, "B")]

# Predicted Values of test set

y_train_names <- all_parameters[str_detect(all_parameters, "y_train_pred")]
y_test_names <- all_parameters[str_detect(all_parameters, "y_test_pred")]

# Group Specific Plots ----------------------------------------------------

rstan::stan_trace(fit, pars = hidden_weights[10:15])
rstan::stan_dens(fit,  pars = hidden_weights[10:15], separate_chains = TRUE)

rstan::get_posterior_mean(fit)

# Prediction of test set

df_post_train <- stan_summary$summary %>% 
  as_tibble() %>% 
  mutate(variable = all_parameters) %>% 
  filter(str_detect(variable, "y_train_pred")) %>% 
  mutate(test_mean = y_train) %>% 
  select(variable, mean, test_mean, everything())

df_post_test <- stan_summary$summary %>% 
  as_tibble() %>% 
  mutate(variable = all_parameters) %>% 
  filter(str_detect(variable, "y_test_pred")) %>% 
  mutate(test_mean = y_test) %>% 
  select(variable, mean, test_mean, everything())

ggplot(df_post_train %>% filter(mean < 5)) + 
  # geom_errorbar(aes(x = mean, ymin = test_mean - sd, ymax = test_mean + sd), alpha=0.3) + 
  geom_point(aes(x = mean, y = test_mean), color="red2", alpha=0.5, size = 2) + 
  geom_abline(slope=1,intercept=0) + 
  theme_bw() + 
  xlab("Predicted") + 
  ylab("Actual") + 
  theme(text = element_text(size=16))

ggplot(df_post_train %>% filter(mean < 5)) + 
  # geom_errorbar(aes(x = mean, ymin = test_mean - sd, ymax = test_mean + sd), alpha=0.3) + 
  geom_point(aes(x = mean, y = test_mean), color="red2", alpha=0.5, size = 2) + 
  geom_abline(slope=1,intercept=0) + 
  theme_bw() + 
  xlab("Predicted") + 
  ylab("Actual") + 
  theme(text = element_text(size=16))

# Text Parsing ---

parse_stan_vars <- function(vars, stan_pattern="W", index_dim=3){

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
    mutate(stan_name = vars) %>% 
    select(stan_name, everything())
  
  return(df_parsed) 
  
}

parse_stan_vars(hidden_weights, "W", 3)
parse_stan_vars(hidden_biases, "B", 2)
parse_stan_vars(y_names, "y_test_pred", 1)


