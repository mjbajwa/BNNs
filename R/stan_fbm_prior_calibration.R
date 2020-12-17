# FBM and STAN gamma equivalence

# FBM Parametrization

FBM_W <- list("GAMMA_WIDTH" = c(0.05, 0.05),
              "GAMMA_ALPHA" = c(0.5, 0.5))

# FBM_W <- list("GAMMA_WIDTH" = c(0.05, 0.05),
#               "GAMMA_ALPHA" = c(0.5, 0.5))

FBM_B <- list("GAMMA_WIDTH" = c(0.05, 0.05),
              "GAMMA_ALPHA" = c(0.5, 0.5))

# FBM to Stan Utils -------------------------------------------------------

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

# Shape = Alpha, Scale = Beta

W_STAN <- fbm_gamma_params_to_stan(FBM_W$GAMMA_WIDTH, FBM_W$GAMMA_ALPHA)
B_STAN <- fbm_gamma_params_to_stan(FBM_B$GAMMA_WIDTH, FBM_B$GAMMA_ALPHA)
W_gamma_shape <- W_STAN$STAN_ALPHA
W_gamma_scale <- W_STAN$STAN_BETA
B_gamma_shape <- B_STAN$STAN_ALPHA
B_gamma_scale <- B_STAN$STAN_BETA

# Example of my parsing of 0.05:5 prior in FBM to Stan

temp <- rgamma(n=10000, shape=W_gamma_shape[1], scale=1/W_gamma_scale[1])
hist(log10(1/sqrt(temp)), col="red2")

# FBM direct

fbm_sdev <- scan("./fbm_logs/x.txt")
hist(log10(fbm_sdev), col="red2")

# Example of another 

example <- fbm_gamma_params_to_stan(fbm_width=0.05, fbm_alpha=2)
temp <- rgamma(n=10000, shape=example$STAN_ALPHA, scale=1/example$STAN_BETA)
hist(log10(1/sqrt(temp)), col="red2")


