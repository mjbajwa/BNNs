#!/bin/bash

# Example FBM - BNN configuration file

net-spec rlog.net 1 8 1 / ih=0.05:0.5 bh=0.05:0.5 ho=x0.05:0.5 bo=100
model-spec rlog.net real 0.05:0.5
net-spec rlog.net
data-spec rlog.net 1 1 / rdata@1:100 . rdata@101:200 .
net-gen rlog.net fix 1 
mc-spec rlog.net repeat 10 sample-noise heatbath hybrid 100:10 0.2
net-mc rlog.net 1

printf "Rejection Rate: \n\n"
net-plt t r rlog.net

printf "Gibbs sampling hyperparameters, and HMC on weights/biases \n\n";
mc-spec rlog.net sample-sigmas heatbath hybrid 1000:10 0.4
net-mc rlog.net 2000

# Plot for analyzing everything is doing what it should...

# net-plt t w3@ rlog.net | graph -n
# net-plt t h3 rlog.net | graph -n -ly
# net-plt t bB rlog.net | graph -n

# Write output to disk

printf "Writing results to disk \n\n";
net-pred itndqQp rlog.net 1000:%10 > results/results.txt;
net-tbl tw1@ rlog.net > results/traces_w1.txt;
net-tbl tw2@ rlog.net > results/traces_w2.txt;
net-tbl tw3@ rlog.net > results/traces_w3.txt;
net-tbl tw4@ rlog.net > results/traces_w4.txt;
net-tbl th1 rlog.net > results/traces_h1.txt;
net-tbl th2 rlog.net > results/traces_h2.txt;
net-tbl th3 rlog.net > results/traces_h3.txt;

printf "Writing step sizes to disk" 
net-stepsizes rlog.net 1000 > results/stepsizes_1000.txt
net-stepsizes rlog.net 1100 > results/stepsizes_1100.txt
net-stepsizes rlog.net 1200 > results/stepsizes_1200.txt
net-stepsizes rlog.net 1300 > results/stepsizes_1300.txt
net-stepsizes rlog.net 1400 > results/stepsizes_1400.txt
net-stepsizes rlog.net 1500 > results/stepsizes_1500.txt
net-stepsizes rlog.net 1600 > results/stepsizes_1600.txt
net-stepsizes rlog.net 1700 > results/stepsizes_1700.txt
net-stepsizes rlog.net 1800 > results/stepsizes_1800.txt
net-stepsizes rlog.net 1900 > results/stepsizes_1900.txt
net-stepsizes rlog.net 2000 > results/stepsizes_2000.txt
