#!/bin/bash
for i in $(seq 1 $2); do 
    echo $i
    sbatch singularity_microgrid_savio.sh $1

done
