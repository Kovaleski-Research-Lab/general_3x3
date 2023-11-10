#!/bin/bash
mpirun --allow-run-as-root --np 48 python3 run.py -lateral_buffer 5.0 -source continuous & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -lateral_buffer 5.0 -source gaussian & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -lateral_buffer 3.0 -source continuous & wait; 
mpirun --allow-run-as-root --np 48 python3 run.py -lateral_buffer 3.0 -source gaussian & wait; 
echo "Ezpz"
