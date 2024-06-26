// Bayesian Neural Network - Non-centered parametrization

data {
  
  // Base inputs
  
  int<lower=0> N; // Number of rows/examples
  int<lower=0> K; // Number of input features
  int<lower=1> h; // Number of hidden layers 
  int G[h]; // Number of neurons per layer (excludes input layer)
  matrix[N, K] X_train; // Input matrix
  vector[N] y_train;    // Target vector
  int<lower=0> N_test; // Number of test examples
  matrix[N_test, K] X_test; // Input matrix for test examples
  
  // Parameters for gamma priors on the *precision* of normal distributions of weights and biases (See R code for more information)
  
  vector[h+1] W_gamma_shape;
  vector[h+1] W_gamma_scale;
  vector[h+1] B_gamma_shape;
  vector[h+1] B_gamma_scale;
  
  // Parameters for gamma priors on the noise level of outputs y (only used if fix_target_noise == 0)
  
  real y_gamma_shape;
  real y_gamma_scale;
  
  // Flag for using Hierarchical priors on weights and biases. If set to 0, then Normal(0, 100) is assumed as the prior.
  
  int<lower=0, upper=1> use_hierarchical_w; 
  int<lower=0, upper=1> use_hierarchical_b;

  // Flag for scaling sdev by number of hidden units in the layer (per Neal 1995) for convergence
  
  int<lower=0, upper=1> infinite_limit[h];
  
  // Flag for fixing target noise. If fix_target_noise = 1, then standard deviation for likelihood is 0.1
  
  int<lower=0, upper=1> fix_target_noise;
  
  // Flag for sampling from prior only
  
  int<lower=0, upper=1> sample_from_prior;
  
}

parameters {
  
  // Weights (W_raw)
  // Hidden layer weight matrices -- h+1 sized array because of additional layer for outputs 
  
  matrix[max(max(G), K), max(max(G), K)] W_raw[h+1]; 
  
  // Intercepts (B_raw)
  
  matrix[max(G), h+1] B_raw; 
  
  // Standard Deviation of likelihood
  
  real log_y_prec;

  // Standard Deviation of Weights and biases
  
  vector[h+1] log_W_prec;
  vector[h+1] log_B_prec; // <lower=1e-6, upper=1e6>[
  
}

transformed parameters {
  
  // Weights (W)

  matrix[max(max(G), K), max(max(G), K)] W[h+1]; 
  
  // Intercepts (b)
  
  matrix[max(G), h+1] B; 

  // Intermediate quantities
  
  matrix[max(G), h] z[N]; // outputs of hidden layers. Array index by oberservation
  vector[N] y_train_pred;
  
  // Exponentiate log variables
  
  real y_prec = exp(log_y_prec);
  vector[h+1] W_prec = exp(log_W_prec);
  vector[h+1] B_prec = exp(log_B_prec);
  
  // ****** Re-parametrization ******

  // Weights

  for(l in 1:(h+1)){
    
    // First hidden layer (connected to inputs)
    
    if(l == 1){ 
      
      for(g_in in 1:K){
        for(g_out in 1:G[l]){
          if(use_hierarchical_w == 1){
            W[l][g_in, g_out] = 1/sqrt(W_prec[l]) * W_raw[l][g_in, g_out]; // normal(0, 1/sqrt(log_W_prec[l])); //
          } else {
            W[l][g_in, g_out] = 100 * W_raw[l][g_in, g_out]; // normal(0, 100);
          }
        }
      }
    
    } else if (l == h+1) {            
      
      // Output layer
      
      for(g_in in 1:G[l-1]) {
        if(use_hierarchical_w == 1){
          if(infinite_limit[l-1] == 1){
            W[l][g_in, 1] = sqrt(1.0/G[l-1]) * 1/sqrt(W_prec[l]) * W_raw[l][g_in, 1]; 
          } else {
            W[l][g_in, 1] = 1/sqrt(W_prec[l]) * W[l][g_in, 1];
          }
        } else {
            W[l][g_in, 1] = 100 * W_raw[l][g_in, 1];
          }
        }
    
    } else {
      
      // All other hidden layers
      
      for(g_in in 1:G[l-1]) {
        for(g_out in 1:G[l]){
          if(use_hierarchical_w == 1){
            if(infinite_limit[l] == 1){
              W[l][g_in, g_out] = sqrt(1.0/G[l]) * 1/sqrt(W_prec[l]) * W_raw[l][g_in, g_out]; 
            } else {
              W[l][g_in, g_out] = 1/sqrt(W_prec[l]) * W_raw[l][g_in, g_out]; 
            }
        } else {
           W[l][g_in, g_out] = 100 * W_raw[l][g_in, g_out];
          }
        }
      }
    }
  }
  
  // Intercepts
  
  for(l in 1:(h+1)){
    if (l == h+1){
      B[1, l] = 100 * B_raw[1, l]; 
    } else {
      for(g in 1:G[l]) {
        if(use_hierarchical_b == 1){
          B[g, l] = 1/sqrt(B_prec[l]) * B_raw[g, l];
        } else {
          B[g, l] = 100 * B_raw[g, l];
        }
      }
    }
  }
  
  // Non-useful weights
  // The extra parameters are added for compatibility with stan data structures. 
  // Priors on these can be ignored for now (non-informative priors assumed by Stan seem to work fine)

  // Weights

  // for(l in 1:(h+1)){
  //   if(l == 1){
  //     for(g_in in (K+1):rows(W[l])){
  //       for(g_out in (G[l]+1):cols(W[l])){
  //         W[l][g_in, g_out] = 100 * W_raw[l][g_in, g_out];
  //       }
  //     }
  //   } else if (l == h+1){
  //     for(g_in in (G[l-1]+1):rows(W[l])) {
  //         W[l][g_in, 1] = 100 * W_raw[l][g_in, 1];
  //       }
  //   } else {
  //     for(g_in in (G[l-1]+1):rows(W[l])) {
  //       for(g_out in (G[l]+1):cols(W[l])){
  //         W[l][g_in, g_out] = 100 * W_raw[l][g_in, g_out];
  //       }
  //     }
  //   }
  // }
  // 
  // // Intercepts
  // 
  // for(l in 1:(h+1)){
  //   if (l == h+1){
  //     for(g in 2:rows(B)){
  //       B[g, l] = 100 * B_raw[g, l];
  //     }
  //   } else {
  //     // For all other layers, index only on useful weights
  //     for(g in (G[l]+1):rows(B)) {
  //       B[g, l] = 100 * B_raw[g, l];
  //     }
  //   }
  // }
  
  // ****** Forward Pass of Neural Network ******

  // Loop over all observations
  for(n in 1:N) {
  
    // calculate for layer = 2:h
    for(l in 1:h){
      
      if(l == 1){  
        // calculate for layer = 1
        for(g in 1:G[l]) {
          z[n][g,1] = tanh(X_train[n, :]*W[l][1:K, g] + B[g, l]); 
        }
      } else {
        // calculate for layer = 2 onwards
        for(g in 1:G[l]){
          z[n][g,l] = tanh(sum(z[n][1:G[l-1], l-1].*W[l][1:G[l-1], g]) + B[g, l]); 
        }
        
      }
    }
    
  }
  
  // Final layer - TODO: Make this multi-output friendly.
  
  for(n in 1:N)
    y_train_pred[n] = sum(z[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
  
}


model {

  // ****** Likelihood function and measurement noise ******
  
  if(sample_from_prior == 1){
    
    if(fix_target_noise == 0){
      y_prec ~ gamma(y_gamma_shape, y_gamma_scale);
    }
    
  } else {
    
    if(fix_target_noise == 1){
      target += normal_lpdf(y_train | y_train_pred, 0.1);
    } else {
      y_prec ~ gamma(y_gamma_shape, y_gamma_scale);
      target += normal_lpdf(y_train | y_train_pred, 1 / sqrt(y_prec));
    }

  }
  
  // ****** Priors ******

  // Weights (USEFUL weights only)
  
  for(l in 1:(h+1)){
    
    // 1st hidden layer
    
    if(l == 1){
      W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]);
      for(g_in in 1:K){
        for(g_out in 1:G[l]){
            W_raw[l][g_in, g_out] ~ normal(0, 1);
          }
        }
      } 
      
    // Output Layer
    
    else if (l == h+1) {
    
      W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]);
      
      for(g_in in 1:G[l-1]) {
        W_raw[l][g_in, 1] ~ normal(0, 1);
      }
    
   // Hidden layers not connected to input/output

    } else {
      
      W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]);
      
      for(g_in in 1:G[l-1]) {
        for(g_out in 1:G[l]){
          W_raw[g_in, g_out] ~ normal(0, 1);
        }
      }
    }
  }

  // Intercepts

  for(l in 1:(h+1)){
    
    if (l == h+1){
      
      // For the output Layer -- only one bias term matters
      
      B_prec[l] ~ gamma(B_gamma_shape[l], B_gamma_scale[l]);
      B_raw[1, l] ~ normal(0, 1); // B[1, l] ~ normal(0, B_sdev[l])
      
    } else {
      
      // For all other layers, index only on useful weights
      
      B_prec[l] ~ gamma(B_gamma_shape[l], B_gamma_scale[l]);
      
      for(g in 1:G[l]) {
        B_raw[g, l] ~ normal(0, 1);
      }
    
    }
  
  }
  
  // ****** Priors on Non-Useful Weights ****** 
  // The extra parameters are added for compatibility with stan data structures. 
  // Priors on these can be ignored for now (non-informative priors assumed by Stan seem to work fine)

  // Weights

  // for(l in 1:(h+1)){
  //   if(l == 1){
  //     for(g_in in (K+1):rows(W[l])){
  //       for(g_out in (G[l]+1):cols(W[l])){
  //         W_raw[l][g_in, g_out] ~ normal(0, 1);
  //       }
  //     }
  //   } else if (l == h+1){
  //     for(g_in in (G[l-1]+1):rows(W[l])) {
  //         W_raw[l][g_in, 1] ~ normal(0, 1);
  //       }
  //   } else {
  //     for(g_in in (G[l-1]+1):rows(W[l])) {
  //       for(g_out in (G[l]+1):cols(W[l])){
  //         W_raw[l][g_in, g_out] ~ normal(0, 1);
  //       }
  //     }
  //   }
  // }

  // // Intercepts
  // 
  // for(l in 1:(h+1)){
  //   if (l == h+1){
  //     for(g in 2:rows(B)){
  //       B_raw[g, l] ~ normal(0, 1);
  //     }
  //   } else {
  //     // For all other layers, index only on useful weights
  //     for(g in (G[l]+ 1):rows(B)) {
  //       B_raw[g, l] ~ normal(0, 1);
  //     }
  //   }
  // }
  
  // ****** Jacobian adjustments due to application of priors on transformed parameters ****** 
  
  // y = exp(x); where y = w_prec, and x = log(w_prec); dy/dx = exp(x) --> log(|dy/dx|) = log(exp(x)) = x
  
  target += log_y_prec;
  
  for(l in 1:(h+1)){
    target += log_W_prec[l];
  }
  
  for(l in 1:h){
    target += log_B_prec[l];
  }

}

generated quantities {
  
  vector[N] y_train_pred_final;
  vector[N_test] y_test_pred;
  vector[N_test] y_test_mean;
  matrix[max(G), h] z_test[N_test]; 

  // Forward pass of neural network until the final layer
  
  for(n in 1:N_test) {
    for(l in 1:h){
      if(l == 1){  
        for(g in 1:G[l]) {
          z_test[n][g,1] = tanh(X_test[n, :]*W[l][1:K, g] + B[g, l]); 
        }
      } else {
        for(g in 1:G[l]){
          z_test[n][g,l] = tanh(sum(z_test[n][1:G[l-1], l-1].*W[l][1:G[l-1], g]) + B[g, l]);
        }
        
      }
    }
    
  }
  
  // Final layer
  
  for(n in 1:N_test){
    
    // Mean prediction
    
    y_test_mean[n] = sum(z_test[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
    
    // Measurement noise
    
    if(fix_target_noise == 1){
      y_test_pred[n] = normal_rng(y_test_mean[n], 0.1);
    } else {
      y_test_pred[n] = normal_rng(y_test_mean[n], sqrt(1 / y_prec));
    }
  }
  
  // Incorporate measurement noise in training predictions as well
  
  for(n in 1:N){
    if(fix_target_noise == 1){
      y_train_pred_final[n] = normal_rng(y_train_pred[n], 0.1);
    } else {
      y_train_pred_final[n] = normal_rng(y_train_pred[n], sqrt(1 / y_prec));
    }
  }
  
}
