#!/bin/bash
t=`date +%Y%m%d-%H%M%S`
(
    echo "experiment: $t"
    echo "args: $@"
    ./src/scripts/experiment.py $@ 
) | tee models/logs/experiment-$t.log
