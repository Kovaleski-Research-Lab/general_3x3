import os
import yaml
import time
import numpy as np
import torch
import meep as mp

# Optimization Libraries
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# CUSTOM LIBRARIES
from meep_utils import simulation

class MeepOracle:
    """
    The Bridge between the abstract Optimizer and the Physics Engine.
    """
    def __init__(self, config):
        self.config = config
        self.sim_count = 0
        
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

        # 5. Compute Figure of Merit (Score)
        score = self._calculate_focusing_score(ex, ey, ez)
        
        print(f"Result: Score = {score:.5f}")
        return score

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


class AutonomousController:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.oracle = MeepOracle(self.config)
        self.n_params = len(self.config['optimization']['parameters'])
        
        # BO State
        self.train_x = []
        self.train_y = []
        
    def initialize_data(self, n_init=5):
        """
        Random Warm-up phase
        """
        print(f"Initializing with {n_init} random simulations...")
        for _ in range(n_init):
            # Generate random normalized parameters [0, 1]
            x_rand = torch.rand(self.n_params)
            y_val = self.oracle.run_sim_and_score(x_rand.numpy())
            
            self.train_x.append(x_rand)
            self.train_y.append(torch.tensor([y_val]))
            
    def run_optimization_loop(self):
        """
        The Main Active Learning Loop
        """
        n_iters = self.config['optimization']['n_iterations']
        
        # Convert lists to tensors
        train_x = torch.stack(self.train_x)
        train_y = torch.stack(self.train_y)
        
        best_value = train_y.max().item()
        
        for i in range(n_iters):
            print(f"\n=== Optimization Iteration {i+1}/{n_iters} ===")
            print(f"Current Best Score: {best_value:.5f}")
            
            # 1. Fit Gaussian Process
            gp = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            # 2. Optimize Acquisition Function (Expected Improvement)
            EI = ExpectedImprovement(gp, best_f=best_value)
            
            # Find the best new point to try
            # bounds are [0,1] because we normalized inside the Oracle
            bounds = torch.stack([torch.zeros(self.n_params), torch.ones(self.n_params)])
            candidate, _ = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20
            )
            
            # 3. Execute Simulation (Oracle)
            new_x = candidate[0] # Tensor
            new_y_val = self.oracle.run_sim_and_score(new_x.numpy())
            
            # 4. Update Data
            train_x = torch.cat([train_x, candidate])
            train_y = torch.cat([train_y, torch.tensor([[new_y_val]])])
            
            # Update best
            if new_y_val > best_value:
                best_value = new_y_val
                print(f"*** New Best Found! {best_value:.5f} ***")

            # Save Checkpoint (Optional)
            if i % 5 == 0:
                torch.save({'x': train_x, 'y': train_y}, 'bo_checkpoint.pt')

if __name__ == "__main__":
    controller = AutonomousController("supplementary/bo_study/config.yaml")
    controller.initialize_data(n_init=3) # Start with 3 random
    controller.run_optimization_loop()