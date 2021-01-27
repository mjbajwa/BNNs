#!/bin/bash

# net-spec rlog2.net 1 1 / bo=0.05:0.5
# net-gen rlog2.net 1000
# net-tbl h1 rlog2.net > x.txt

net-spec rlog2.net 1 8 1 / ih=0.05:0.5 bh=0.05:0.5 ho=x0.05:0.5 bo=100
model-spec rlog2.net real 0.05:0.5
net-spec rlog2.net
data-spec rlog2.net 1 1 / rdata@1:100 . rdata@101:200 .
net-gen rlog2.net 10000
net-tbl h1 rlog2.net > final.txt
