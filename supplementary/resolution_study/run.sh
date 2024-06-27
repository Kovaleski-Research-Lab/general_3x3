#!/bin/bash

resolutions=(70 80 90 100 110 120)

for i in "${!resolutions[@]}"; do
    resolution=${resolutions[$i]}
    mpirun --allow-run-as-root -np 56 python3 main.py -config config.yaml -resolution $resolution 
done
