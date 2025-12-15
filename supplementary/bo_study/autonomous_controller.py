import os
import yaml
import time
import numpy as np
import torch
import meep as mp
import matplotlib.pyplot as plt
import shutil
import csv
import datetime
from torch.quasirandom import SobolEngine

# Optimization Libraries
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# CUSTOM LIBRARIES
from meep_utils import simulation

# =============================================================================
#   HELPER: DIRECTORY MANAGEMENT
# =============================================================================
def setup_run_directory(base_path="../../results", config_path="opt_config.yaml"):
    """Creates a unique timestamped folder and backs up config."""
    
    # Create timestamp (e.g., run_20251212_143005)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_path, f"run_{timestamp}")
    
    # Create subfolders
    plots_dir = os.path.join(run_dir, "plots")
    sims_dir = os.path.join(run_dir, "sim_visuals")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(sims_dir, exist_ok=True)
    
    # Backup Config
    shutil.copy(config_path, os.path.join(run_dir, "config_snapshot.yaml"))
    
    # Initialize CSV Log
    csv_path = os.path.join(run_dir, "optimization_log.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: Iteration, Score, Param1, Param2, ...
        writer.writerow(['Iteration', 'Score', 'Radius', 'Height', 'Index', 'Best_So_Far'])
        
    print(f"\n[System] Run Initialized: {run_dir}")
    return run_dir, plots_dir, sims_dir, csv_path

# =============================================================================
#   MEEP ORACLE
# =============================================================================
class MeepOracle:
    """
    The Bridge between the abstract Optimizer and the Physics Engine.
    """
    def __init__(self, config, sims_dir):
        self.config = config
        self.sims_dir = sims_dir
        self.sim_count = 0
        
        if not os.path.exists("debug_plots"):
            os.makedirs("debug_plots")
        
    def _parameter_mapper(self, x_normalized):
        """
        Maps the optimizer's input vector (0-1 range) to physical units.
        Assumes x is [radius, height, index]
        """
        bounds = self.config['optimization']['parameters']
        
        # Unpack bounds
        r_min, r_max = bounds[0]['bounds']
        h_min, h_max = bounds[1]['bounds']
        n_min, n_max = bounds[2]['bounds']
        
        # De-normalize
        r = x_normalized[0] * (r_max - r_min) + r_min
        h = x_normalized[1] * (h_max - h_min) + h_min
        n = x_normalized[2] * (n_max - n_min) + n_min
        
        return float(r), float(h), float(n)

    def run_sim_and_score(self, x_in):
        """
        1. Builds Sim
        2. Runs Sim
        3. Extracts Data (In-Memory)
        4. Calculates Score
        """
        self.sim_count += 1
        print(f"\n--- Starting Sim #{self.sim_count} ---")
        
        # 1. Map Parameters
        # Assuming grid_size=1 for now. If 2x2, we would construct lists of length 4.
        radius, height, n_index = self._parameter_mapper(x_in)
        
        print(f"Params: Radius={radius:.3f}, Height={height:.3f}, Index={n_index:.3f}")
        
        # Construct lists expected by simulation.build_sim
        radii = [radius]
        heights = [height]
        epsilons = [n_index**2] # Convert n to epsilon
        
        # 2. Build Simulation
        # We pass the lists directly, bypassing the random generation logic in your old script
        sim, dft_obj, flux_obj, _ = simulation.build_sim(
            self.config, 
            radii=radii, 
            heights=heights, 
            epsilons=epsilons
        )
        
        # 3. Run Simulation
        # Using a simple condition for now
        sim.run(until=self.config['simulation']['until'])
        
        # 4. In-Memory Data Extraction (The Efficiency Hack)
        # Instead of writing H5, we pull the arrays directly.
        # Note: MEEP returns data as complex numbers if fields are complex
        
        # We only grab the center wavelength for optimization speed
        target_wl = self.config['source']['wavelength']
        # Find index of target_wl in the flux object (assuming center freq)
        freq_idx = 0 
        
        # Extract fields directly to RAM
        # Shape: [x, y, z] complex128
        ex = sim.get_dft_array(dft_obj, mp.Ex, freq_idx)
        ey = sim.get_dft_array(dft_obj, mp.Ey, freq_idx)
        ez = sim.get_dft_array(dft_obj, mp.Ez, freq_idx)
        
        # Clean up MEEP object to free memory immediately
        sim.reset_meep()
        del sim

        # Compute Figure of Merit (Score)
        #score = self._calculate_focusing_score(ex, ey, ez)
        #print(f"Result: Score = {score:.5f}")
        #return score
        
        # Calculate Score (Contrast Ratio)
        return self._calculate_contrast_score(ex, ey, ez, radius, height, n_index)

    def _calculate_contrast_score(self, ex, ey, ez, r, h, n):
        intensity = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2
        nx, ny, nz = intensity.shape
        
        # Focus on Center Slice
        center_idx = nz // 2
        slice_plane = intensity[:, :, center_idx]
        
        max_val = np.max(slice_plane)
        mean_val = np.mean(slice_plane)
        
        # Avoid divide by zero
        if mean_val == 0: return 0.0
        
        # Metric: Contrast (Max/Mean - 1)
        score = (max_val / mean_val) - 1.0
        
        # Save Visualization
        plt.figure(figsize=(5, 4))
        plt.imshow(slice_plane, cmap='inferno')
        plt.colorbar(label='Intensity')
        plt.title(f"Sim #{self.sim_count}\nR={r:.2f} H={h:.2f} n={n:.2f}\nScore: {score:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.sims_dir, f"sim_{self.sim_count:03d}.png"))
        plt.close()
        
        return float(score)

    def _calculate_focusing_score(self, ex, ey, ez):
        """
        Calculates intensity at the target point / total power.
        """
        # Calculate Total Intensity |E|^2
        intensity = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2
        
        # Get dimensions
        nx, ny, nz = intensity.shape
        
        # Find the "Focal Plane" (usually the top-most Z slice)
        # Or you can use specific coordinates if you map indices to physical space
        focal_plane_idx = -5 # Look 5 pixels from the top boundary
        focal_plane = intensity[:, :, focal_plane_idx]
        
        # Find intensity at center
        mid_x, mid_y = nx // 2, ny // 2
        # A small 3x3 window average is more robust than a single pixel
        spot_intensity = np.mean(focal_plane[mid_x-1:mid_x+2, mid_y-1:mid_y+2])
        
        # Normalize by total power in that plane (simpler than flux for now)
        total_power = np.sum(focal_plane)
        
        if total_power == 0: return 0.0
        
        return spot_intensity / total_power

# =============================================================================
#   The Controller
# =============================================================================
class AutonomousController:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup Logging
        self.run_dir, self.plots_dir, self.sims_dir, self.csv_path = setup_run_directory(config_path=config_path)
        
        self.oracle = MeepOracle(self.config, self.sims_dir)
        self.n_params = len(self.config['optimization']['parameters'])
        self.n_init = self.config['optimization']['n_init']
        self.method = self.config['optimization']['method']
        
        self.train_x = []
        self.train_y = []
        self.history_best = []
        
    def _log_to_csv(self, iteration, score, params, best_so_far):
        # Map normalized params back to real units for readable logs
        real_params = self.oracle._parameter_mapper(params)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, score, *real_params, best_so_far])
        
    def initialize_data(self):
        """
        Random Warm-up phase
        """
        print(f"Initializing with {self.n_init} random simulations...")
        for i in range(self.n_init):
            # Generate random normalized parameters [0, 1]
            x_rand = torch.rand(self.n_params, dtype=torch.double)
            y_val = self.oracle.run_sim_and_score(x_rand.numpy())
            
            self.train_x.append(x_rand)
            self.train_y.append(torch.tensor([y_val], dtype=torch.double))
            
            # Update History
            current_best = max([y.item() for y in self.train_y])
            self.history_best.append(current_best)
            
            self._log_to_csv(i, y_val, x_rand.numpy(), current_best)
            
    def run_optimization_loop(self):
        """
        The Main Active Learning Loop
        """
        n_iters = self.config['optimization']['n_iterations']
        
        if self.method == 'bo': # bayesian optimization
            # Convert lists to tensors
            train_x = torch.stack(self.train_x)
            train_y = torch.stack(self.train_y)
            best_value = train_y.max().item()  
        elif self.method in ['random-search', 'sobol']: # start at ground zero
            train_x = torch.empty((0, self.n_params), dtype=torch.double)
            train_y = torch.empty((0, 1), dtype=torch.double) 
            best_value = 0.0
            if self.method == 'sobol': # quasi-random
                sobol = SobolEngine(dimension=self.n_params, scramble=True)
                sobol_samples = sobol.draw(n_iters).to(dtype=torch.double)
            
        for i in range(n_iters):
            iteration_id = len(self.history_best)
            print(f"\n=== Optimization Iteration {i+1}/{n_iters} ===")
            print(f"Current Best Score: {best_value:.5f}")
            
            if self.method == 'bo': # bayesian optimization
                
                # 1. Fit Gaussian Process
                gp = SingleTaskGP(train_x, train_y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                
                # 2. Optimize Acquisition Function (Expected Improvement)
                EI = LogExpectedImprovement(gp, best_f=best_value)
                
                # Find the best new point to try
                # bounds are [0,1] because we normalized inside the Oracle
                bounds = torch.stack([
                    torch.zeros(self.n_params, dtype=torch.double), 
                    torch.ones(self.n_params, dtype=torch.double)
                ])
                new_x, _ = optimize_acqf(
                    EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20
                )
                
                # VISUALIZE ACQUISITION
                self._plot_acquisition_slice(gp, EI, new_x[0], iteration_id)
                
                new_x = new_x[0]
                
            elif self.method == 'random-search':
                new_x = torch.rand(self.n_params, dtype=torch.double)
                
            elif self.method == 'sobol':
                new_x = sobol_samples[i]
                
            # Execute Simulation (Oracle)
            new_y_val = self.oracle.run_sim_and_score(new_x.numpy())
            
            # Update Data
            train_x = torch.cat([train_x, new_x.unsqueeze(0)])
            train_y = torch.cat([train_y, torch.tensor([[new_y_val]], dtype=torch.double)])
            
            # logging
            current_best = train_y.max().item()
            self.history_best.append(current_best)
            self._log_to_csv(iteration_id, new_y_val, new_x.numpy(), current_best)
            
            # update Performance Plots
            self._plot_performance_trace()

            # Save Checkpoint
            if i % 5 == 0:
                save_path = os.path.join(self.run_dir, 'checkpoint.pt')
                torch.save({'x': train_x, 'y': train_y}, save_path)

    def _plot_performance_trace(self):
        """Plots the history of Score and Best Score."""
        scores = [y.item() for y in self.train_y] # All scores
        best_scores = self.history_best # Best so far
        
        plt.figure(figsize=(10, 5))
        plt.plot(scores, 'o-', alpha=0.4, label='Individual Sim')
        plt.plot(best_scores, 'r-', linewidth=2, label='Best Found')
        plt.xlabel('Iteration')
        plt.ylabel('Contrast Score')
        if self.method == 'bo':
            plt.title('Bayesian Optimization Progress')
        elif self.method == 'random-search':
            plt.title('Random Search Progress')
        elif self.method == 'sobol':
            plt.title('Quasi-Random Search (Sobol) Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, "trace_score.png"))
        plt.close()

    def _plot_acquisition_slice(self, gp, acq_func, candidate, iter_id):
        """
        Plots a 2D slice of the Acquisition Function (Expected Improvement).
        Slice is taken at the 'candidate' point for the 3rd dimension.
        We plot Radius (x) vs Height (y), holding Index fixed.
        """
        # Create a grid [0,1] x [0,1]
        n_grid = 50
        x = torch.linspace(0, 1, n_grid, dtype=torch.double)
        y = torch.linspace(0, 1, n_grid, dtype=torch.double)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # Prepare test points (Shape: [2500, 3])
        # We vary Radius (dim 0) and Height (dim 1), fix Index (dim 2)
        fixed_index = candidate[2].item()
        
        grid_points = torch.stack([X.flatten(), Y.flatten(), torch.full_like(X.flatten(), fixed_index)], dim=1)
        
        # Evaluate Acquisition Function
        with torch.no_grad():
            acq_values = acq_func(grid_points.unsqueeze(1)) # Shape [2500]
            acq_grid = acq_values.reshape(n_grid, n_grid)
            
        # Plot
        plt.figure(figsize=(7, 6))
        # Use pcolormesh
        plt.contourf(X.numpy(), Y.numpy(), acq_grid.numpy(), levels=20, cmap='viridis')
        plt.colorbar(label='Expected Improvement (EI)')
        
        # Mark the chosen candidate
        plt.scatter(candidate[0].item(), candidate[1].item(), c='red', marker='*', s=150, label='Next Sim')
        
        plt.xlabel('Normalized Radius')
        plt.ylabel('Normalized Height')
        plt.title(f'Decision Surface (Iter {iter_id})\nSlice at Norm. Index = {fixed_index:.2f}')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, f"acq_func_iter_{iter_id:03d}.png"))
        plt.close()

if __name__ == "__main__":
    controller = AutonomousController("supplementary/bo_study/config.yaml")
    controller.initialize_data()
    controller.run_optimization_loop()