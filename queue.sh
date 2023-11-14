#!/bin/bash
mpirun --allow-run-as-root --np 48 python3 run.py -idx 0 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 1 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 2 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 3 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 4 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 5 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 6 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 7 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 8 & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -idx 9 & wait; 
echo "Ezpz"
