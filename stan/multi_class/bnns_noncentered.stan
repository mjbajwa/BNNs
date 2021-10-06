// Bayesian Neural Network - Non-centered parametrization

data {
  
  // Base inputs
  
  int<lower=0> N; // Number of rows/examples
  int<lower=0> K; // Number of input features
  int<lower=1> h; // Number of hidden layers 
  int G[h]; // Number of neurons per layer (excludes input layer)
  matrix[N, K] X_train; // Input matrix
  int y_train[N];    // Target vector
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
  
  // int<lower=0, upper=1> infinite_limit[h];
  
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
  
  # real<lower=0> y_prec;

  // Standard Deviation of Weights and biases
  
  vector<lower=0>[h+1] W_prec;
  vector<lower=0>[h+1] B_prec; // <lower=1e-6, upper=1e6>
  
  // Automatic Relevance Determination
  
  vector<lower=0>[K] ard_prec_raw;
  
}

transformed parameters {
  
  // Weights (W)

  matrix[max(max(G), K), max(max(G), K)] W[h+1]; 
  
  // Intercepts (b)
  
  matrix[max(G), h+1] B; 

  // Intermediate quantities
  
  matrix[max(G), h] z[N]; // outputs of hidden layers. Array index by oberservation
  matrix[N, 3] y_train_mean;
  
  // ARD parameters
  
  vector[K] ard_prec;
  
  // ****** Re-parametrization ******

  // Weights and ARD parameters

  for(l in 1:(h+1)){
    
    // First hidden layer (connected to inputs)
    
    if(l == 1){ 
      
      // ARD parameters
      
      for(g_in in 1:K){
        ard_prec[g_in] = 1/(0.5)*W_prec[l]*ard_prec_raw[g_in];
      }
      
      // Weights
      
      for(g_in in 1:K){
        for(g_out in 1:G[l]){
          if(use_hierarchical_w == 1){
            W[l][g_in, g_out] = 1/sqrt(ard_prec[g_in]) * W_raw[l][g_in, g_out]; // normal(0, 1/sqrt(W_prec[l])); //
          } else {
            W[l][g_in, g_out] = 100 * W_raw[l][g_in, g_out]; // normal(0, 100);
          }
        }
      }
    
    } else if (l == h+1) {            
      
      // Output layer
      
      for(g_in in 1:G[l-1]) {
        if(use_hierarchical_w == 1){
            W[l][g_in, 1] = 1/sqrt(W_prec[l]) * W_raw[l][g_in, 1];
        } else {
            W[l][g_in, 1] = 100 * W_raw[l][g_in, 1];
          }
        }
    
    } else {
      
      // All other hidden layers
      
      for(g_in in 1:G[l-1]) {
        for(g_out in 1:G[l]){
          if(use_hierarchical_w == 1){
            W[l][g_in, g_out] = 1/sqrt(W_prec[l]) * W_raw[l][g_in, g_out]; 
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
      for(g in 1:3){
        B[g, l] = 1/sqrt(B_prec[l]) * B_raw[g, l];
      }
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

  // // Weights
  // 
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
  
  for(n in 1:N){
    for(o in 1:3){
      y_train_mean[n, o] = sum(z[n][1:G[h], h].*W[h+1][1:G[h], o]) + B[o, h+1];
    }
  }
}


model {
  
  for (n in 1:N)
    y_train[n] ~ categorical_logit(y_train_mean[n]'); // implicitly categorical(y | softmax(alpha))

  // ****** Likelihood function and measurement noise ******
  
  // if(sample_from_prior == 1){
  //   
  //   if(fix_target_noise == 0){
  //     y_prec ~ gamma(y_gamma_shape, y_gamma_scale);
  //   }
  //   
  // } else {
  //   
  //   if(fix_target_noise == 1){
  //     target += normal_lpdf(y_train | y_train_pred, 0.1);
  //   } else {
  //     y_prec ~ gamma(y_gamma_shape, y_gamma_scale);
  //     target += normal_lpdf(y_train | y_train_pred, 1 / sqrt(y_prec));
  //   }
  // 
  // }
  
  // ****** Priors ******

  // Weights (USEFUL weights only)
  
  for(l in 1:(h+1)){
    
    // 1st hidden layer
    
    if(l == 1){
      
      W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]);
      
      for(g_in in 1:K){
        
        ard_prec_raw[g_in] ~ gamma(0.5, 1);
        
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
      
      for (o in 1:3){
        B_raw[o, l] ~ normal(0, 1);
      }
      
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

  // // Weights
  // 
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
  // 
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
    
}

generated quantities {
  
   // Declare variables for categorical predictions
  
  matrix[N_test, 3] y_test_mean;
  int y_test_pred[N_test];
  matrix[max(G), h] z_test[N_test];
  int y_train_pred[N];
  
  // Declare variables to store probability estimates

  matrix[N, 3] y_train_prob;
  matrix[N_test, 3] y_test_prob;

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
    
    for(o in 1:3){
      y_test_mean[n, o] = sum(z_test[n][1:G[h], h].*W[h+1][1:G[h], o]) + B[o, h+1];
    }
    
    y_test_pred[n] = categorical_logit_rng(y_test_mean[n]');
  
  }
  
  // Re-compute training prediction
  
  for(n in 1:N){
    y_train_pred[n] = categorical_logit_rng(y_train_mean[n]');
  }
  
  // Probability for train and test 
  
  for(n in 1:N){
    
    y_train_prob[n, 1] = exp(y_train_mean[n, 1])/sum(exp(y_train_mean[n]));
    y_train_prob[n, 2] = exp(y_train_mean[n, 2])/sum(exp(y_train_mean[n]));
    y_train_prob[n, 3] = 1 - y_train_prob[n, 1] - y_train_prob[n, 2];
    
  }
  
  for(n in 1:N_test){
    
    y_test_prob[n, 1] = exp(y_test_mean[n, 1])/sum(exp(y_test_mean[n]));
    y_test_prob[n, 2] = exp(y_test_mean[n, 2])/sum(exp(y_test_mean[n]));
    y_test_prob[n, 3] = 1 - y_test_prob[n, 1] - y_test_prob[n, 2];
    
  }
  
}
