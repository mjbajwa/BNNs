// Bayesian Neural Network 
// To modify activation functions, change "tanh" on line 43, 48 to a bounded function of your choice
// TODO: multiple outputs not yet supported.

data {
  int<lower=0> N; // Input Data rows
  int<lower=0> K; // input features
  int<lower=1> h; // number of hidden layers 
  int G[h]; // number of neurons per layer
  matrix[N, K] X_train; // Input matrix
  vector[N] y_train;    // Target vector
  int<lower=0> N_test; // number of test variables
  matrix[N_test, K] X_test; // Input matrix
}

parameters {
  
  // Weights
  matrix[max(max(G), K), max(max(G), K)] W[h+1]; // Hidden layer weight matrices -- H+1 size because of additional layer for outputs 
  
  // Intercepts
  matrix[max(G), h+1] B; // matrix for hidden layer intercepts
  
  // Variance
  real<lower=0> sigma;
  
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
  
  // Priors on inputs (Control hyperparameters of normal using inputs)
  
  for(l in 1:(h+1)){
    if(l == 1){
      for(g in 1:K){
        W[l][g, l] ~ normal(0, 1); 
      }
    } else {
      for(g in 1:G[l-1]) { // max(max(G), K)
        W[l][g, l] ~ normal(0, 1);
      }
    }
  }
  
  // Intercepts
  
  for(l in 1:(h+1)){
    if(l == 1){
      for(g in 1:G[l]){
        B[g, l] ~ normal(0, 1); 
      }
    } else {
      for(g in 1:max(G)) {
        B[g, l] ~ normal(0, 1);
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
          z_test[n][g,l] = tanh(sum(z_test[n][1:G[l-1], l-1].*W[l][1:G[l-1], g]) + B[g, l]); // 
          
        }
        
      }
    }
    
  }
  
  // Final Layer
  
  for(n in 1:N_test)
    y_test_pred[n] = sum(z_test[n][1:G[h], h].*W[h+1][1:G[h], 1]) + B[1, h+1];
  
}

