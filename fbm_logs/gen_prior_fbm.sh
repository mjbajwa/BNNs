#!/bin/bash

# net-spec prior_log_"$1" 1 1 / bo=0.05:0.5
# net-gen prior_log_"$1" 1000
# net-tbl h1 prior_log_"$1" > x.txt

net-spec prior_log_"$1".net 1 8 1 / ih=0.05:0.5 bh=0.05:0.5 ho=x0.05:0.5 bo=100
rand-seed prior_log_"$1".net $2
model-spec prior_log_"$1".net real 0.05:0.5
data-spec prior_log_"$1".net 1 1 / rdata@1:100 . rdata@101:200 .
net-gen prior_log_"$1".net 2000

printf "Writing results to disk \n\n";
net-pred itndqQp prior_log_"$1".net 1000:%10 > prior/chain_"$1"/results.txt
net-tbl tw1@ prior_log_"$1".net > prior/chain_"$1"/traces_w1.txt;
net-tbl tw2@ prior_log_"$1".net > prior/chain_"$1"/traces_w2.txt;
net-tbl tw3@ prior_log_"$1".net > prior/chain_"$1"/traces_w3.txt;
net-tbl tw4@ prior_log_"$1".net > prior/chain_"$1"/traces_w4.txt;
net-tbl th1 prior_log_"$1".net > prior/chain_"$1"/traces_h1.txt;
net-tbl th2 prior_log_"$1".net > prior/chain_"$1"/traces_h2.txt;
net-tbl th3 prior_log_"$1".net > prior/chain_"$1"/traces_h3.txt;
net-tbl tn@ prior_log_"$1".net > prior/chain_"$1"/traces_y_sdev.txt