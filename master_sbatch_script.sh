#!/bin/bash
for i in $(seq 1 $1); do 
    echo $i
    sbatch singularity_microgrid.sh
done