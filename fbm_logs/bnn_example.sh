#!/bin/bash

# Example FBM - BNN configuration file

net-spec rlog_"$1".net 1 8 1 / ih=0.05:0.5 bh=0.05:0.5 ho=x0.05:0.5 bo=100
rand-seed rlog_"$1".net $2
model-spec rlog_"$1".net real 0.05:0.5
net-spec rlog_"$1".net
data-spec rlog_"$1".net 1 1 / rdata@1:100 . rdata@101:200 .
net-gen rlog_"$1".net fix 1 
mc-spec rlog_"$1".net repeat 10 sample-noise heatbath hybrid 100:10 0.2
net-mc rlog_"$1".net 1

printf "Rejection Rate: \n\n"
net-plt t r rlog_"$1".net

printf "Gibbs sampling hyperparameters, and HMC on weights/biases \n\n";
mc-spec rlog_"$1".net sample-sigmas heatbath hybrid 1000:10 0.4
net-mc rlog_"$1".net 2000

# Plot for analyzing everything is doing what it should...

# net-plt t w3@ rlog_"$1".net | graph -n
# net-plt t h3 rlog_"$1".net | graph -n -ly
# net-plt t bB rlog_"$1".net | graph -n

# Write output to disk

mkdir -p chain_"$1"/results/

printf "Writing results to disk \n\n";
net-pred itndqQp rlog_"$1".net 1000:%10 > chain_"$1"/results/results.txt;
net-tbl tiG@ rlog_"$1".net 1000:%10 > chain_"$1"/results/test_target_samples.txt
net-tbl tw1@ rlog_"$1".net > chain_"$1"/results/traces_w1.txt;
net-tbl tw2@ rlog_"$1".net > chain_"$1"/results/traces_w2.txt;
net-tbl tw3@ rlog_"$1".net > chain_"$1"/results/traces_w3.txt;
net-tbl tw4@ rlog_"$1".net > chain_"$1"/results/traces_w4.txt;
net-tbl th1 rlog_"$1".net > chain_"$1"/results/traces_h1.txt;
net-tbl th2 rlog_"$1".net > chain_"$1"/results/traces_h2.txt;
net-tbl th3 rlog_"$1".net > chain_"$1"/results/traces_h3.txt;
net-tbl tn@ rlog_"$1".net > chain_"$1"/results/traces_y_sdev.txt

printf "Writing step sizes to disk" 
net-stepsizes rlog_"$1".net 1000 > chain_"$1"/results/stepsizes_1000.txt
net-stepsizes rlog_"$1".net 1100 > chain_"$1"/results/stepsizes_1100.txt
net-stepsizes rlog_"$1".net 1200 > chain_"$1"/results/stepsizes_1200.txt
net-stepsizes rlog_"$1".net 1300 > chain_"$1"/results/stepsizes_1300.txt
net-stepsizes rlog_"$1".net 1400 > chain_"$1"/results/stepsizes_1400.txt
net-stepsizes rlog_"$1".net 1500 > chain_"$1"/results/stepsizes_1500.txt
net-stepsizes rlog_"$1".net 1600 > chain_"$1"/results/stepsizes_1600.txt
net-stepsizes rlog_"$1".net 1700 > chain_"$1"/results/stepsizes_1700.txt
net-stepsizes rlog_"$1".net 1800 > chain_"$1"/results/stepsizes_1800.txt
net-stepsizes rlog_"$1".net 1900 > chain  _"$1"/results/stepsizes_1900.txt
net-stepsizes rlog_"$1".net 2000 > chain_"$1"/results/stepsizes_2000.txt
