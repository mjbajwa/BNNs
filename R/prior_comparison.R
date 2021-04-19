priors <- function(width = 0.05, alpha = 0.5, label="Experiment 1"){
  
  FBM_W <- list("GAMMA_WIDTH" = rep(width, 1),
                "GAMMA_ALPHA" = rep(alpha, 1))
  
  FBM_B <- list("GAMMA_WIDTH" = rep(width, 1),
                "GAMMA_ALPHA" = rep(alpha, 1))
  
  FBM_Y <- list("GAMMA_WIDTH" = rep(width, 1),
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

df <- priors(0.5, 0.75, "Experiment 1") %>% 
  bind_rows(priors(0.5, 0.5, "Experiment 2")) %>% 
  bind_rows(priors(0.1, 5, "Experiment 3")) %>% 
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

# New function (March 23rd) -----------------------------------------------

fbm_gamma_params_to_stan <- function(fbm_width, fbm_alpha, N = NULL) {
  
  # TODO: check with Prof Neal this re-parametrization is correct.
  
  # if(is.null(N)){
  #   mean_precision = 1 / (fbm_width ^ 2)
  # } else if(fbm_alpha < 2 & !is.null(N)) {
  #   mean_precision = (N^(2/fbm_alpha)) * 1 / (fbm_width ^ 2)
  # } else if (fbm_alpha == 2 & !is.null(N)) {
  #   mean_precision = (N * log(N) / (fbm_width ^ 2))
  # } else if (fbm_alpha > 2 & !is.null(N)) {
  #   mean_precision = (N * (fbm_alpha / fbm_alpha - 2)) / (fbm_width ^ 2)
  # }
  
  mean_precision = (N) / (fbm_width^2)
  
  # Convert to Stan parametrization
  
  stan_alpha = fbm_alpha / 2
  stan_beta = stan_alpha / mean_precision
  
  output = list("STAN_ALPHA" = stan_alpha,
                "STAN_BETA" = stan_beta)
  
  return(output)
  
}

args <- fbm_gamma_params_to_stan(c(0.05, 0.05), c(0.5, 0.5), 1)
temp <- rgamma(n = 1000, shape = args[[1]], scale = 1 / args[[2]])
hist(log10(1 / sqrt(temp)), col = "red2", main = "Weights (log10 sdev)")
