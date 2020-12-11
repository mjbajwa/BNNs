// Bayesian Neural Network 
// To modify activation functions, change "tanh" on line 43, 48 to a bounded function of your choice
// TODO: multiple outputs not yet supported.

data {
  int<lower=0> N; // Input Data examples
  int<lower=0> K; // Input features
  int<lower=1> h; // Number of hidden layers 
  int G[h]; // Number of neurons per layer
  matrix[N, K] X_train; // Input matrix
  vector[N] y_train;    // Target vector
  int<lower=0> N_test; // Number of test examples
  matrix[N_test, K] X_test; // Input matrix for test examples
  
  # Hyperpriors on the Weights!


}

parameters {
  
  // Weights (W)
  matrix[max(max(G), K), max(max(G), K)] W[h+1]; // Hidden layer weight matrices -- H+1 size because of additional layer for outputs 
  
  // Intercepts (b)
  matrix[max(G), h+1] B; // matrix for hidden layer intercepts
  
  // Variance of likelihood
  real<lower=0> sigma;
  
  // Hyperparameter on weights of each layer
  // vector[h] W_sdev;
  
  // Hyperparameters for gamma priors on the variance of each group
  // vector[h] shape_W_sdev; 
  // vector[h] precision_W_sdev;
  
}

transformed parameters {
  
  matrix[max(G), h] z[N]; // outputs of hidden layer. Each observation produces Gxh of these. Array index per observation.
  vector[N] y_train_pred;
  
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
          z[n][g,l] = tanh(sum(z[n][1:G[l-1], l-1].*W[l][1:G[l-1], g]) + B[g, l]); // 
          
        }
        
      }
    }
    
  }
  
  // Final Layer
  
  for(n in 1:N)
    y_train_pred[n] = sum(z[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
  
}

model {
  
  // Likelihood function
  
  y_train ~ normal(y_train_pred, sigma);
  
  // Prior on sigma
  
  sigma ~ cauchy(0, 1);
  
  // ****** Priors on Useful Weights ******
  
  // Priors on weights (USEFUL weights only)
  
  for(l in 1:(h+1)){
    if(l == 1){ // Hidden layer connected to Input Layer
      for(g_in in 1:K){
        for(g_out in 1:G[l]){
          W[l][g_in, g_out] ~ normal(0, 1); 
        }
      }
    } else if (l == h+1) { // Output Layer 
      for(g_in in 1:G[l-1]) { 
          W[l][g_in, 1] ~ normal(0, 1); 
        }
    } else {
      for(g_in in 1:G[l-1]) { // All hidden layers not connected to input/output
        for(g_out in 1:G[l-1]){
          W[l][g_in, g_out] ~ normal(0, 1); 
        }
      }
    }
  }
  
  // Priors on intercepts (USEFUL intercepts only for each column of the B matrix)
  
  // for(l in 1:(h+1)){
  //   if (l == h+1){ 
  //     // For the output Layer -- only one bias term matters
  //       B[1, l] ~ normal(0, 1);
  //   } else { 
  //     // For all other layers, index only on useful weights
  //     for(g in 1:G[l]) {
  //       B[g, l] ~ normal(0, 1);
  //     }
  //   }
  // }
  // 
  // // ****** Priors on Non-Useful Weights ****** // Consider removing for simplicity
  // 
  // // Weights
  // 
  // for(l in 1:(h+1)){
  //   if(l == 1){
  //     for(g_in in (K+1):rows(W[l])){
  //       for(g_out in (G[l]+1):cols(W[l])){
  //         W[l][g_in, g_out] ~ uniform(-2, 2);
  //       }
  //     }
  //   } else if (l == h+1){
  //     for(g_in in (G[l-1]+1):rows(W[l])) { 
  //         W[l][g_in, 1] ~ uniform(-2, 2); 
  //       }
  //   } else {
  //     for(g_in in (G[l-1]+1):rows(W[l])) {
  //       for(g_out in (G[l]+1):cols(W[l])){
  //         W[l][g_in, g_out] ~ uniform(-2, 2);
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
  //       B[g, l] ~ uniform(-2, 2);
  //     }
  //   } else { 
  //     // For all other layers, index only on useful weights
  //     for(g in (G[l]+1):rows(B)) {
  //       B[g, l] ~ uniform(-2, 2);
  //     }
  //   }
  // }
    
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
          z_test[n][g,l] = tanh(sum(z_test[n][1:G[l-1], l-1].*W[l][1:G[l-1], g]) + B[g, l]); // 
        }
        
      }
    }
    
  }
  
  // Final Layer
  
  for(n in 1:N_test)
    y_test_pred[n] = sum(z_test[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
  
}

