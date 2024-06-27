import sys
import os
import time
import pickle


def run():

    cores_list = list(range(16, 113, 8))
    time_list = []

    for cores in cores_list:

        print(f"\nBeginning simulation on {cores} cores...\n")
        time_start = time.time()

        os.system(f"mpirun --allow-run-as-root --oversubscribe -np {cores} python3 ../../main.py -config ../../configs/config.yaml -idx 0")

        time_total = (time.time() - time_start) / 60

        print(f"\nTotal time for {cores} cores is {time_total} minutes.\n")

        time_list.append(time_total)

    results = [cores_list, time_list]

    return results


if __name__=="__main__":

    results = run()

    filename = "core_study_60_round2.pkl"
    pickle.dump(results,open(filename,"wb"))
