// Bayesian Neural Network 
// TODO: 1) Add varying number of neurons in each layer
// TODO: 2) Add choice of priors by layer (prior over functions) - controlled by hyperparameters.

functions {
  real sigmoid_activation(real x) {
    return (1/(1 + exp(-x)));
  }
}

data {
  int<lower=0> N; // Data rows
  int<lower=0> K; // input features
  int<lower=1> G; // number of neurons
  int<lower=1> h; // number of hidden layers for basic FFNN
  matrix[N, K] X; // Input matrix
  vector[N] y;    // y vector
}

parameters {
  
  // Weights
  matrix[K, G] W_initial; // input weights
  matrix[G, G] W[h-1]; // Hidden layer weight matrices
  vector[G] W_final; // final weights connecting last hidden layer to output
  
  // Intercepts
  matrix[G, h] B; // matrix of vectors for hidden unit intercepts
  real B_final; // final B
  
  // Variance
  real<lower=0> sigma;
  
}

transformed parameters {
  
  matrix[G, h] z[N]; // outputs of hidden layer. Each observation produces Gxh of these. Array index per observation.
  vector[N] y_pred;
  
  // Loop over all observations
  for(n in 1:N) {
    
    // calculate for layer = 1
    for(g in 1:G) {
      z[n][g,1] = tanh(X[n, :]*W_initial[:, g] + B[g,1]); // TODO: activation function
    }

    // calculate for layer = 2:h
    for(l in 2:h){
      for(g in 1:G){
        z[n][g,l] = tanh(sum(z[n][:, l-1].*W[l-1][:, g]) + B[g,l]); // TODO: activation function // WRONG
      }
    }
  
  }
  
  // Final Layer
  
  for(n in 1:N)
    y_pred[n] = sum(z[n][:, h].*W_final) + B_final;
  
}

model {
  
  // Likelihood function
    
  y ~ normal(y_pred, sigma);
  
  // Prior on sigma
  
  sigma ~ cauchy(0, 1);
  
  // Priors on inputs
  
  for(k in 1:K)
    for(g in 1:G)
      W_initial[k, g] ~ normal(0, 1);
  
  for(l in 1:(h-1)){
    for(g in 1:G){
      for(s in 1:G){
        W[l][g, s] ~ normal(0, 1);
      }
    }
  }
  
  for(g in 1:G){
    W_final[g] ~ normal(0, 1);
  }
  
}

