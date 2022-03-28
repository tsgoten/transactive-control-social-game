#!/bin/bash
for i in $(seq 1 $2); do 
    echo $i
    sbatch singularity_microgrid.sh $1
    sleep 3
done