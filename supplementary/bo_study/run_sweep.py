import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import yaml
import argparse

from autonomous_controller import run_meep_worker

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_ground_truth_sweep(param='radius'):
    
    if param == 'radius':
        sweep_param = np.linspace(0.1, 0.45, 30)
        param_2 = 1.02 # height
        param_3 = 3.48 # refractive index
    elif param == 'height':
        sweep_param = np.linspace(0.75, 1.25, 30)
        param_2 = 0.2 # radius
        param_3 = 3.48 # refractive index
    elif param == 'index':
        sweep_param = np.linspace(2.0, 4.0, 30)
        param_2 = 0.2 # radius
        param_3 = 1.02 # height
    
    scores = []
    
    print(f"Starting Linear Sweep on {len(sweep_param)} points...")
    
    # 2. Run Sweep (Serial or Parallel)
    # Using the same worker logic to ensure consistency
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    
    config = load_config('supplementary/bo_study/config.yaml') # Make sure to load your YAML config here
    
    for i in sweep_param:
        print(f"Simulating {param}: {i:.4f}...", end="", flush=True)
    
        if param == 'radius':
            param_list = [i, param_2, param_3]
        elif param == 'height':
            param_list = [param_2, i, param_3]
        elif param == 'index':
            param_list = [param_2, param_3, i]
        
        # Launch Worker
        p = ctx.Process(target=run_meep_worker, args=(config, param_list, queue))
        p.start()
        result = queue.get()
        p.join()
        
        if isinstance(result, Exception):
            print(f" Error: {result}")
            scores.append(0.0)
        else:
            print(f" Score: {result:.4f}")
            scores.append(result)

    # 3. Plot the Truth
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_param, scores, 'b-o', linewidth=2, label='Simulation Data')
    plt.xlabel(f"{param} (microns)")
    plt.ylabel("Bandpass Score")
    plt.title("The Ground Truth: Physics Landscape")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"ground_truth_sweep_{param}.png")
    print(f"Sweep complete. Saved to ground_truth_sweep_{param}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param', required=True)
    args = vars(parser.parse_args())
    run_ground_truth_sweep(args['param'])