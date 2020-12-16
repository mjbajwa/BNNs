#!/bin/bash

# Example FBM - BNN configuration file

net-spec rlog.net 1 8 1 / ih=0.05:0.5 bh=0.05:0.5 ho=0.05:0.5 bo=100
model-spec rlog.net real 0.05:0.5 
net-spec rlog.net
model-spec rlog.net
data-spec rlog.net 1 1 / rdata@1:100 . rdata@101:200 .
net-gen rlog.net fix 0.5
mc-spec rlog.net repeat 10 sample-noise heatbath hybrid 100:10 0.2
net-mc rlog.net 1
mc-spec rlog.net sample-sigmas heathbath hybrid 1000:10 0.4
net-mc rlog.net 400
net-display rlog.net
net-pred itn rlog.net 1 > example.txt
