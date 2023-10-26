import subprocess

def run_script():
    
    neighbors_list = [i for i in range(500)]

    for index in neighbors_list:
        print(f"index = {index}")
        command = f"mpirun -np 40 python3 run_sim.py -neighbor_index {index} -resim False"
 
        stdout_file = f"output_{index}.txt"
        stderr_file = f"error_{index}.txt"
    
        with open(stdout_file, "w") as stdout_f, open(stderr_file, "w") as stderr_f:
            
            script = subprocess.run(command, shell=True, stdout=stdout_f, stderr=stderr_f)

    if script.returncode == 0:
        print(f"Finished running {index+1} sims")
        print("Output:", script.stdout)
    else:
        print("Error occurred.")
        print("Error message:", script.stderr)

if __name__=="__main__":
    
    run_script()
