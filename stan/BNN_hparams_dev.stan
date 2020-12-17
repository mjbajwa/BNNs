// Bayesian Neural Network 
// 1. To modify activation functions, change "tanh" on line 43, 48 to a bounded function of your choice
// 2. Prior structure on weights and biases: W[L, i, j] ~ normal(0, sig[L]^2); 1/sig^2 ~ gamma(shape, rate). Single level hierarchy assumed.
// 3. Standard deviations not yet scaled using the number of weights in a given layer per Neal (1995). 

data {
  
  // Basic inputs 
  
  int<lower=0> N; // Input Data examples
  int<lower=0> K; // Input features
  int<lower=1> h; // Number of hidden layers 
  int G[h]; // Number of neurons per layer
  matrix[N, K] X_train; // Input matrix
  vector[N] y_train;    // Target vector
  int<lower=0> N_test; // Number of test examples
  matrix[N_test, K] X_test; // Input matrix for test examples
  
  // Parameters for gamma hyperpriors on the *precision* of normal distributions
  
  vector[h+1] W_gamma_shape;
  vector[h+1] W_gamma_scale;
  // vector[h+1] B_gamma_shape;
  // vector[h+1] B_gamma_scale;
  
  // Flag for using Hierarchical (helpful for debugging)
  
  int<lower = 0, upper = 1> use_hierarchical; // 0 = use N(0,1), 1 = use N(0, sdev), prec = 1/(sdev^2), prec ~ Gamma(shape, scale)
  
}

parameters {
  
  // Weights (W)
  matrix[max(max(G), K), max(max(G), K)] W[h+1]; // Hidden layer weight matrices -- H+1 size because of additional layer for outputs 
  
  // Intercepts (b)
  matrix[max(G), h+1] B; // matrix for hidden layer intercepts
  
  // Standard Deviation of likelihood
  real<lower=0> sigma;

  // Standard Deviation of Weights
  vector<lower=0>[h+1] W_sdev;
  
}

transformed parameters {
  
  // Define precision terms (1/sigma^2)
  vector[h+1] W_prec;

  // Intermediate quantities
  matrix[max(G), h] z[N]; // outputs of hidden layer. Each observation produces Gxh of these. Array index per observation.
  vector[N] y_train_pred;
  
  // *** Forward Pass of Neural Network ***
  
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
  
  // Final Layer. TODO: Make this multi-output friendly.
  
  for(n in 1:N)
    y_train_pred[n] = sum(z[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
    
  // *** Precision transformation ***
  
  // Weight parameters (useful weights only)
  
  for(l in 1:(h+1)){
    W_prec[l] = 1/(W_sdev[l]^2);
  }
    
}


model {

  // ****** Likelihood function and measurement noise ******
  
  y_train ~ normal(y_train_pred, sigma);
  sigma ~ normal(0, 1);
  
  // ***** Apply proper priors on all sdev and precision variables *****
  
  // ****** Direct Priors on Useful Weights ******
  
  // Priors on weights (USEFUL weights only)
  // For each layer, precisions are sampled from a common prior distribution (gamma)
  
  for(l in 1:(h+1)){
    if(l == 1){ // Hidden layer connected to Input Layer
      for(g_in in 1:K){
        for(g_out in 1:G[l]){
          W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]); 
          if(use_hierarchical == 1){
            W[l][g_in, g_out] ~ normal(0, W_sdev[l]); // * (1/sqrt(G[l]))); 
          } else {
            W[l][g_in, g_out] ~ normal(0, 1);
          }
        }
      }
    } else if (l == h+1) { // Output Layer 
      for(g_in in 1:G[l-1]) {
        W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]);
        if(use_hierarchical == 1){
            W[l][g_in, 1] ~ normal(0, W_sdev[l]); // * (1/sqrt(G[l]))); 
          } else {
             W[l][g_in, 1] ~ normal(0, 1);
          }
        }
    } else {
      for(g_in in 1:G[l-1]) { // All hidden layers not connected to input/output
        for(g_out in 1:G[l]){
          W_prec[l] ~ gamma(W_gamma_shape[l], W_gamma_scale[l]); 
          if(use_hierarchical == 1){
            W[l][g_in, g_out] ~ normal(0, W_sdev[l]); // * (1/sqrt(G[l]))); 
          } else {
            W[l][g_in, g_out] ~ normal(0, 1);
          }
        }
      }
    }
  }
  
  // Priors on intercepts (USEFUL intercepts only for each column of the B matrix)
  
  for(l in 1:(h+1)){
    if (l == h+1){
      // For the output Layer -- only one bias term matters
      // B_sdev[1, l] ~ gamma(B_gamma_shape[l], B_gamma_scale[l]);
      if(use_hierarchical == 1){
        B[1, l] ~ normal(0, 1); // normal(0, B_sdev[1, l]);
      } else {
        B[1, l] ~ normal(0, 1);
      }
    } else {
      // For all other layers, index only on useful weights
      for(g in 1:G[l]) {
      // B_sdev[g, l] ~ gamma(B_gamma_shape[l], B_gamma_scale[l]);
        if(use_hierarchical == 1){
          B[g, l] ~ normal(0, 1); // normal(0, B_sdev[g, l]);
        } else {
          B[g, l] ~ normal(0, 1);
        }
      }
    }
  }
  
  // ****** Jacobian adjustments ******  
  
  // Required because we're applying a prior to (prec = 1/(sdev^2)).
  
  if(use_hierarchical == 1){
    for(l in 1:(h+1)){
      target += log(2) - 3*log(W_sdev[l]);
    }
  }

  // ****** Priors on Non-Useful Weights ****** // Keep these hard-coded for now

  // Weights

  for(l in 1:(h+1)){
    if(l == 1){
      for(g_in in (K+1):rows(W[l])){
        for(g_out in (G[l]+1):cols(W[l])){
          W[l][g_in, g_out] ~ normal(0, 10);
        }
      }
    } else if (l == h+1){
      for(g_in in (G[l-1]+1):rows(W[l])) {
          W[l][g_in, 1] ~ normal(0, 10);
        }
    } else {
      for(g_in in (G[l-1]+1):rows(W[l])) {
        for(g_out in (G[l]+1):cols(W[l])){
          W[l][g_in, g_out] ~ normal(0, 10);
        }
      }
    }
  }

  // Intercepts

  for(l in 1:(h+1)){
    if (l == h+1){
      for(g in 2:rows(B)){
        B[g, l] ~ normal(0, 10);
      }
    } else {
      // For all other layers, index only on useful weights
      for(g in (G[l]+1):rows(B)) {
        B[g, l] ~  normal(0, 10);
      }
    }
  }
    
}

generated quantities {
  
  vector[N_test] y_test_pred;
  matrix[max(G), h] z_test[N_test]; 

  // Loop over all observations
  for(n in 1:N_test) {
    
    // calculate for layer = 2:h
    for(l in 1:h){
      if(l == 1){  
        // calculate for layer = 1
        for(g in 1:G[l]) {
          z_test[n][g,1] = tanh(X_test[n, :]*W[l][1:K, g] + B[g, l]); 
        }
      } else {
        // calculate for layer = 2 onwards
        for(g in 1:G[l]){
          z_test[n][g,l] = tanh(sum(z_test[n][1:G[l-1], l-1].*W[l][1:G[l-1], g]) + B[g, l]);
        }
        
      }
    }
    
  }
  
  // Final Layer
  
  for(n in 1:N_test)
    y_test_pred[n] = sum(z_test[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
  
}

