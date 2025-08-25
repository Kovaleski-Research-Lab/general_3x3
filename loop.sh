#!/bin/bash
for i in {76..1500}
do
	mpirun -np 32 python run.py $i
done
