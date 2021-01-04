priors <- function(alpha = 0.5, label="Experiment 1"){
  
  FBM_W <- list("GAMMA_WIDTH" = rep(0.05, 1),
                "GAMMA_ALPHA" = rep(alpha, 1))
  
  FBM_B <- list("GAMMA_WIDTH" = rep(0.05, 1),
                "GAMMA_ALPHA" = rep(alpha, 1))
  
  FBM_Y <- list("GAMMA_WIDTH" = rep(0.05, 1),
                "GAMMA_ALPHA" = rep(alpha, 1))
  
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
  precision_b <- rgamma(n=1000, shape=B_gamma_shape[1], scale=1/B_gamma_scale[1])
  precision_y <- rgamma(n=1000, shape=Y_gamma_shape[1], scale=1/Y_gamma_scale[1])
  
  df <- tibble(
    weights = log10(1/sqrt(precision_w)), 
    biases = log10(1/sqrt(precision_b)), 
    y_noise = log10(1/sqrt(precision_y)),
    label=label
  )
  
  return(df)
  
}

df <- priors(alpha = 0.5, label = "Experiment 1") %>% 
  bind_rows(priors(alpha=1, label="Experiment 2")) %>% 
  bind_rows(priors(alpha=5, label="Experiment 3")) %>% 
  pivot_longer(!label, names_to = "vars")

ggplot(df) + 
  geom_density(aes(x = value, color = label, fill = label), alpha = 0.4, size = 1.2) + 
  scale_color_viridis(discrete=T) + 
  scale_fill_viridis(discrete=T) + 
  theme_bw() + 
  facet_wrap(vars~., scales = "free") +
  theme(text = element_text(size = 16),
        legend.position = "bottom",
        legend.title = element_blank()) + 
  xlab("")