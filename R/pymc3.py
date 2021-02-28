import pymc3 as pm
import theano
import numpy as np
import pandas as pd
import lasagne
import os
import datetime

floatX = theano.config.floatX
RANDOM_SEED = 42
ITER = 10000  # TODO: add to config


def load_data():
    df = pd.read_table("./data/rdata", header=None, delim_whitespace=True)
    df.columns = ["X", "Y"]
    df["index"] = np.where(df.index < 100, "Train", "Test")

    X_train = np.array(df.loc[df["index"] == "Train", "X"]).reshape(-1, 1).astype(floatX)
    Y_train = np.array(df.loc[df["index"] == "Train", "Y"]).astype(floatX)
    X_test = np.array(df.loc[df["index"] == "Test", "X"]).reshape(-1, 1).astype(floatX)
    Y_test = np.array(df.loc[df["index"] == "Test", "Y"]).astype(floatX)

    return X_train, Y_train, X_test, Y_test


# Centered parametrization

def pymc3_centered(X_train=None, Y_train=None):
    # TODO: add to config

    n_hidden = [8]

    # Initialize random weights between each layer

    init_w_ih = np.zeros((X_train.shape[1], n_hidden[0])).astype(floatX)
    init_w_ho = np.zeros((n_hidden[0], 1)).astype(floatX)
    init_b_h = np.zeros((n_hidden[0],)).astype(floatX)
    init_b_o = np.zeros((1,)).astype(floatX)

    # Initialize Model

    with pm.Model() as neural_network:
        bnn_input = theano.shared(X_train, name="bnn_input")
        bnn_output = theano.shared(Y_train, name="bnn_output")

        # ==== Hyperparameter Prior Definition (Precision)

        prec_w_ih = pm.Gamma("W_prec_ih", alpha=0.025, beta=0.000625, testval=1)
        prec_w_ho = pm.Gamma("W_prec_ho", alpha=0.025, beta=0.000625, testval=1)
        prec_b_h = pm.Gamma("B_prec_h", alpha=0.025, beta=0.000625, testval=1)
        prec_target = pm.Gamma("y_prec", alpha=0.025, beta=0.000625, testval=1)

        # ==== Low-level parameters Prior Definition

        # Weights from input to hidden layer

        weights_ih = pm.Normal("w_ih", mu=0, tau=prec_w_ih, shape=(X_train.shape[1], n_hidden[0]),
                               testval=init_w_ih)

        # Weights from hidden layer to output

        weights_ho = pm.Normal("w_ho", mu=0, sigma=1 / np.sqrt(prec_w_ho) * (1 / np.sqrt(8)), shape=(n_hidden[0], 1),
                               testval=init_w_ho)

        # Biases of hidden layer

        biases_h = pm.Normal("b_h", mu=0, tau=prec_b_h, shape=(n_hidden[0],), testval=init_b_h)

        # Biases of output layer

        biases_o = pm.Normal("b_o", mu=0, sigma=100, shape=(1,), testval=init_b_o)

        # ==== Forward Pass of the Neural Network

        # Build neural-network using Lasagne (keras equivalent)

        act_in = lasagne.layers.InputLayer(X_train.shape, input_var=bnn_input)

        act_h = lasagne.layers.DenseLayer(incoming=act_in,
                                          num_units=n_hidden[0],
                                          W=weights_ih,
                                          b=biases_h,
                                          nonlinearity=lasagne.nonlinearities.tanh)

        act_out = lasagne.layers.DenseLayer(incoming=act_h,
                                            num_units=1,
                                            W=weights_ho,
                                            b=biases_o)

        net_out = lasagne.layers.get_output(act_out)

        # Sample priors first (Likelihood added later due to bug in Lasagne-PyMC3-Theano interaction)

        # prior_checks = pm.sample_prior_predictive(samples=500, random_seed=RANDOM_SEED)

        output = pm.Normal("output", mu=net_out.flatten(ndim=1), tau=prec_target, observed=bnn_output)

        # Sampling steps

        step1 = pm.NUTS([prec_w_ih, prec_w_ho, prec_b_h, prec_target])
        step2 = pm.NUTS([weights_ih, weights_ho, biases_h, biases_o])
        steps = pm.CompoundStep([step1, step2])

    return neural_network


# # Non-centered parametrization

def pymc3_noncentered(X_train=None, Y_train=None):

    n_hidden = [8]

    # Initialize random weights between each layer

    init_w_ih = np.zeros((X_train.shape[1], n_hidden[0])).astype(floatX)
    init_w_ho = np.zeros((n_hidden[0], 1)).astype(floatX)
    init_b_h = np.zeros((n_hidden[0], )).astype(floatX)
    init_b_o = np.zeros((1, )).astype(floatX)

    with pm.Model() as neural_network:

        bnn_input = theano.shared(X_train)
        bnn_output = theano.shared(Y_train)

        # ==== Hyperparameter Prior Definition (Precision)

        prec_w_ih = pm.Gamma("W_prec_ih", alpha=0.025, beta=0.000625, testval = 0.0001)
        prec_w_ho = pm.Gamma("W_prec_ho", alpha=0.025, beta=0.000625, testval = 0.0001)
        prec_b_h = pm.Gamma("B_prec_h", alpha=0.025, beta=0.000625, testval = 1)
        prec_target = pm.Gamma("y_prec", alpha=0.025, beta=0.000625, testval = 1)

        # ==== Re-parametrized

        weights_ih_raw = pm.Normal("w_ih_raw", mu=0, sigma=1, shape=(X_train.shape[1], n_hidden[0]), testval=init_w_ih)

        # Weights from hidden layer to output

        weights_ho_raw = pm.Normal("w_ho_raw", mu=0, sigma=1, shape=(n_hidden[0], 1), testval=init_w_ho)

        # Biases of hidden layer

        biases_h_raw = pm.Normal("b_h_raw", mu=0, sigma=1, shape=(n_hidden[0], ), testval=init_b_h)

        # Weights from input to hidden layer

        weights_ih = 1/np.sqrt(prec_w_ho)*weights_ih_raw

        # Weights from hidden layer to output

        weights_ho = (1/np.sqrt(prec_w_ho))*(1/np.sqrt(8))*weights_ho_raw

        # Biases of hidden layer

        biases_h = 1/np.sqrt(prec_b_h)*biases_h_raw

        # Biases of output layer

        biases_o = pm.Normal("b_o", mu=0, sigma=100, shape=(1,), testval=init_b_o)

        # ==== Forward Pass of the Neural Network

        # Build neural-network using Lasagne (keras equivalent)

        act_in = lasagne.layers.InputLayer(X_train.shape, input_var=bnn_input)

        act_h = lasagne.layers.DenseLayer(incoming=act_in,
                                          num_units=n_hidden[0],
                                          W=weights_ih,
                                          b=biases_h,
                                          nonlinearity=lasagne.nonlinearities.tanh)

        act_out = lasagne.layers.DenseLayer(incoming=act_h,
                                            num_units=1,
                                            W=weights_ho,
                                            b=biases_o)

        net_out = lasagne.layers.get_output(act_out)

        # Add likelihood function

        output = pm.Normal("output", net_out.flatten(), sigma=1/np.sqrt(prec_target), observed=bnn_output)

        # Sampling methods

        step1 = pm.NUTS([prec_w_ih, prec_w_ho, prec_b_h, prec_target])
        step2 = pm.NUTS([weights_ih, weights_ho, biases_h, biases_o])
        steps = pm.CompoundStep([step1, step2])

    return neural_network

def pymc3_inference(neural_network):

    with neural_network:
        trace = pm.sample(draws=ITER,
                          chains=4,
                          cores=4,
                          # step=steps,
                          return_inferencedata=False)

    # Train predictions

    with neural_network:
        ppc_train = pm.sample_posterior_predictive(
            trace, var_names=["output"], random_seed=42, samples=ITER,
        )

    # Test predictions

    bnn_input.set_value(X_test)
    bnn_output.set_value(Y_test)

    with neural_network:
        ppc_test = pm.sample_posterior_predictive(
            trace, var_names=["output"], random_seed=42, samples=ITER,
        )

    return ppc_train, ppc_test

def pymc3_organize_results(ppc_train, ppc_test):

    df_predictions_train = pd.DataFrame({

        "inputs": X_train.flatten(),
        "targets": Y_train.flatten(),
        "mean": ppc_train["output"][1000:].mean(axis=0),
        "median": np.quantile(ppc_train["output"][1000:], 0.5, axis=0),
        "q1": np.quantile(ppc_train["output"][1000:], 0.01, axis=0),
        "q10": np.quantile(ppc_train["output"][1000:], 0.10, axis=0),
        "q90": np.quantile(ppc_train["output"][1000:], 0.90, axis=0),
        "q99": np.quantile(ppc_train["output"][1000:], 0.99, axis=0),
        "label": "train"

    })

    df_predictions_test = pd.DataFrame({

        "inputs": X_test.flatten(),
        "targets": Y_test.flatten(),
        "mean": ppc_test["output"][1000:].mean(axis=0),
        "median": np.quantile(ppc_test["output"][1000:], 0.5, axis=0),
        "q1": np.quantile(ppc_test["output"][1000:], 0.01, axis=0),
        "q10": np.quantile(ppc_test["output"][1000:], 0.10, axis=0),
        "q90": np.quantile(ppc_test["output"][1000:], 0.90, axis=0),
        "q99": np.quantile(ppc_test["output"][1000:], 0.99, axis=0),
        "label": "test"

    })

    df_predictions = pd.concat([df_predictions_train, df_predictions_test]).reset_index()

    df_traces = pm.trace_to_dataframe(trace,
                                      chains=[0, 1, 2, 3],
                                      varnames=["W_prec_ih", "W_prec_ho", "B_prec_h", "y_prec", "w_ih_raw", "w_ho_raw",
                                                "b_h_raw", "b_o"])

    df_traces["id"] = df_traces.index

    df_traces["trace"] = np.where(np.logical_and(df_traces["id"] >= 0, df_traces["id"] < ITER), 1,
                                  np.where(np.logical_and(df_traces["id"] >= ITER, df_traces["id"] < 2 * ITER), 2,
                                           np.where(np.logical_and(df_traces["id"] >= 2 * ITER, df_traces["id"] < 3 * ITER),
                                                    3,
                                                    np.where(np.logical_and(df_traces["id"] >= 3 * ITER,
                                                                            df_traces["id"] < 4 * ITER), 4, 0))))

    return df_predictions, df_traces


def pymc3_write_to_disk(df_predictions, df_traces):

    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    new_dir = os.path.join("output/", "pymc3_" + time)
    os.mkdir(new_dir)
    df_traces.to_feather(f"{new_dir}/df_traces.feather")
    df_predictions.drop(f"index", axis=1).to_feather(f"{new_dir}/df_predictions.feather")
    print(new_dir)


