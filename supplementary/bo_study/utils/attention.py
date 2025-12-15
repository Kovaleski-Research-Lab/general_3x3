import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class AcquisitionTracker:
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.params_config = config['optimization']['parameters']
        self.param_names = [p['name'] for p in self.params_config]
        self.n_params = len(self.param_names)
        
        # We will store the history as a list of 2D arrays (Iter x Grid_Size) for each param
        self.history = {name: [] for name in self.param_names}
        self.sampled_points = {name: [] for name in self.param_names}
        
        self.grid_size = 100
        self.grids = [torch.linspace(0, 1, self.grid_size) for _ in range(self.n_params)]

    def log_step(self, gp, acq_func, best_params_norm):
        """
        Calculates the 1D marginal acquisition slice for each parameter,
        holding the others fixed at the 'best_params_norm' values.
        """
        gp.eval()
        
        for i, name in enumerate(self.param_names):
            # 1. Create a probe line for parameter 'i'
            # Start with the 'best' vector repeated 100 times
            probe_points = best_params_norm.repeat(self.grid_size, 1)
            
            # Replace column 'i' with the linear grid (0 to 1)
            probe_points[:, i] = self.grids[i]
            
            # 2. Evaluate Acquisition Function on this line
            with torch.no_grad():
                # BoTorch expects (N, 1, D)
                vals = acq_func(probe_points.unsqueeze(1))
                
                # Normalize values to 0-1 range for Visualization consistency
                # (We care about RELATIVE interest at this step)
                vals = vals.detach().squeeze()
                if vals.max() > vals.min():
                    vals = (vals - vals.min()) / (vals.max() - vals.min())
                
                self.history[name].append(vals.numpy())
                
            # Store the actual value sampled at this step for the overlay scatter
            # (We assume the best_params_norm passed in IS the new candidate, or we track separately)
            # For this viz, let's track the 'best so far' which is passed in
            self.sampled_points[name].append(best_params_norm[i].item())

    def plot_timeline(self):
        """Generates the Heatmap Timeline."""
        fig, axes = plt.subplots(self.n_params, 1, figsize=(10, 8), sharex=True)
        
        iteration_count = len(list(self.history.values())[0])
        x_extent = [0, iteration_count, 0, 1] # [Left, Right, Bottom, Top]

        for i, name in enumerate(self.param_names):
            ax = axes[i]
            
            # Convert list of arrays to matrix (Rows=Iterations)
            # We transpose so X-axis is Iteration, Y-axis is Parameter
            matrix = np.array(self.history[name]).T 
            
            # Plot Heatmap
            im = ax.imshow(matrix, aspect='auto', extent=x_extent, origin='lower', cmap='viridis')
            
            # Overlay the trajectory (The path the optimizer actually took)
            # Note: Ideally, pass the *new candidate* to log_step to plot the 'choice'
            # If we plot 'best_so_far', we see the convergence line.
            trajectory = self.sampled_points[name]
            ax.plot(range(iteration_count), trajectory, 'r-', linewidth=1, alpha=0.5, label='Best So Far')
            
            # Styling
            p_min, p_max = self.params_config[i]['bounds']
            ax.set_ylabel(f"{name}\n({p_min} - {p_max})", fontsize=10)
            ax.set_yticks([0.1, 0.5, 0.9])
            ax.set_yticklabels([f"{p_min + 0.1*(p_max-p_min):.2f}", 
                                f"{p_min + 0.5*(p_max-p_min):.2f}", 
                                f"{p_min + 0.9*(p_max-p_min):.2f}"])
            
            if i == 0:
                ax.set_title("Optimization Attention Timeline (Bright = High Interest)", fontsize=14)
        
        axes[-1].set_xlabel("Iteration Number", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "acquisition_timeline.png"), dpi=200)
        plt.close()
        print("Timeline plot saved.")