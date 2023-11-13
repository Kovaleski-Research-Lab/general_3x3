#!/bin/bash
mpirun --allow-run-as-root --np 48 python3 run.py -idx 9 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 10 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 11 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 12 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 13 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 14 & wait; 
echo "Ezpz"
