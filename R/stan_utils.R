
read_input_data <- function(input_type = "fbm_example|power_plant") {
  
  # Load data
  
  if (input_type == "fbm_example") {
    
    df <- data.frame(read.table("./data/rdata", header = FALSE))
    colnames(df) <- c("V1", "Y")
    target_col <- "Y"
    
  } else if (input_type == "power_plant") {
    
    df <- read.csv("./data/power-plant.csv", header = TRUE)
    colnames(df) <- c("V1", "V2", "V3", "V4", "Y")
    target_col <- "Y"
    
  } else {
    
    break
    
  }
  
  return(df)
  
}


fbm_gamma_params_to_stan <- function(fbm_width, fbm_alpha) {
  
  # TODO: check with Prof Neal this re-parametrization is correct.
  
  mean_precision = 1 / (fbm_width ^ 2)
  stan_alpha = fbm_alpha / 2
  stan_beta = stan_alpha / mean_precision
  
  output = list("STAN_ALPHA" = stan_alpha,
                "STAN_BETA" = stan_beta)
  
  return(output)
  
}


parse_stan_vars <- function(vars,
                            stan_pattern = "W",
                            index_dim = 3,
                            column_names = c("layer", "incoming_neuron", "outgoing_neuron")) {
  
  #' Parses stan variables
  
  temp <- str_replace(vars, str_c(stan_pattern, "\\["), "") %>%
    str_replace(str_c(stan_pattern, "_raw", "\\["), "") %>% 
    str_replace("\\]", "") %>%
    str_split(",", n = index_dim) %>%
    lapply(function(x) {
      c(x)
    })
  
  parsed_outputs <-
    matrix(data = 0,
           nrow = length(temp),
           ncol = index_dim)
  
  for (i in 1:length(temp)) {
    parsed_outputs[i, ] <- as.numeric(temp[i] %>% unlist())
  }
  
  df_parsed <- as_tibble(parsed_outputs)
  names(df_parsed) <- column_names
  df_parsed <- df_parsed %>%
    mutate(stan_var_name = vars) %>%
    select(stan_var_name, everything())
  
  return(df_parsed)
  
}


markov_chain_samples <- function(stan_fit,
                                 var,
                                 n_chains = 4,
                                 burn_in = 1000,
                                 iters = 2000) {
  
  # Create empty dataframe to save results
  
  df_chains <- as_tibble(matrix(
    data = 0,
    nrow = iters,
    ncol = n_chains
  ))
  names(df_chains) <-
    as.vector(unlist(lapply(as.list(1:n_chains), function(x) {
      paste("chain_", x, sep = "")
    })))
  
  # Loop through each chain and extract the samples
  
  for (chain in 1:n_chains) {
    temp <- stan_fit@sim$samples[[chain]]
    parameter_samples <- temp[[var]]
    
    df_chains[as.character(paste("chain_", chain, sep = ""))] <-
      parameter_samples
    
  }
  
  # Calculate across chain averages (TODO: might also want to do within chain averages for R_hat)
  
  df_chains["time_index"] = 1:nrow(df_chains)
  
  df_chain_summary <- df_chains %>%
    reshape2::melt(id.vars = "time_index") %>%
    group_by(time_index) %>%
    summarize(mean_chains = mean(value),
              sdev_chains = sd(value)) %>%
    ungroup()
  
  # Join the summary with final
  
  df_chains["var"] = var
  df_chains["stationary"] = ifelse(1:nrow(df_chains) > burn_in, T, F)
  
  df_chains <- df_chains %>%
    left_join(df_chain_summary, by = "time_index")
  
  return(df_chains)
  
}


mcmc_trace_plot <- function(df_mcmc_param, var, burn_in = 1000, min_time = 0) {
  
  df_plot <- df_mcmc_param %>%
    select(time_index, contains("chain_")) %>%
    reshape2::melt(id.vars = "time_index",) %>%
    mutate(stationary = ifelse(time_index > burn_in, T, F))
  
  ggplot(df_plot %>% filter(time_index > min_time)) +
    geom_point(aes(
      x = time_index,
      y = value,
      color = variable,
      alpha = stationary
    ),
    size = 0.5) +
    scale_alpha_manual(values = c(0.20, 1)) +
    geom_vline(xintercept = burn_in,
               linetype = 2) +
    theme_bw() +
    ggtitle(str_c(unique(df_mcmc_param$var))) +
    xlab("") +
    ylab("") +
    theme(text = element_text(size = 20),
          legend.position = "none")
  
}


mcmc_density_plot <- function(df_mcmc_param, var, burn_in = 1000, min_time = 0) {
  
  df_plot <- df_mcmc_param %>%
    select(time_index, contains("chain_")) %>%
    reshape2::melt(id.vars = "time_index",) %>%
    mutate(stationary = ifelse(time_index > burn_in, T, F))
  
  ggplot(df_plot %>% filter(time_index > burn_in)) +
    geom_density(aes(
      x = value,
      color = variable,
      alpha = 0.5
    ),
    size = 0.5) +
    theme_bw() +
    ggtitle(str_c(unique(df_mcmc_param$var))) +
    xlab("") +
    ylab("") +
    theme(text = element_text(size = 20),
          legend.position = "none")
  
}