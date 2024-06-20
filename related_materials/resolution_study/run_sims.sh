#!/bin/bash

resolutions=(35 45 55)

for i in "${!resolutions[@]}"; do
    res=${resolutions[$i]}
    mpirun --allow-run-as-root -np 56 python3 main.py -config ../../configs/config.yaml -res $res -idx 0
done
