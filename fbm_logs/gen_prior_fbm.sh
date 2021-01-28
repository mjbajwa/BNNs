#!/bin/bash

# net-spec rlog2.net 1 1 / bo=0.05:0.5
# net-gen rlog2.net 1000
# net-tbl h1 rlog2.net > x.txt

net-spec rlog2.net 1 8 1 / ih=0.05:0.5 bh=0.05:0.5 ho=x0.05:0.5 bo=100
model-spec rlog2.net real 0.05:0.5
data-spec rlog2.net 1 1 / rdata@1:100 . rdata@101:200 .
net-gen rlog2.net 2000

printf "Writing results to disk \n\n";
net-pred itndqQp rlog2.net 1000:%10 > results_priors/results.txt
net-tbl tw1@ rlog2.net > results_priors/traces_w1.txt;
net-tbl tw2@ rlog2.net > results_priors/traces_w2.txt;
net-tbl tw3@ rlog2.net > results_priors/traces_w3.txt;
net-tbl tw4@ rlog2.net > results_priors/traces_w4.txt;
net-tbl th1 rlog2.net > results_priors/traces_h1.txt;
net-tbl th2 rlog2.net > results_priors/traces_h2.txt;
net-tbl th3 rlog2.net > results_priors/traces_h3.txt;
net-tbl tn@ rlog2.net > results_priors/traces_y_sdev.txt

