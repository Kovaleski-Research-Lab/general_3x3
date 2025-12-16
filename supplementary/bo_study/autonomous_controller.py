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
import gc
import multiprocessing
import cmath

# Optimization Libraries
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel

# CUSTOM LIBRARIES
from meep_utils import simulation
from utils.plotting import OptimizationAnimator
from utils.attention import AcquisitionTracker

# =============================================================================
#   HELPER: WORKER FOR MULTIPROCESSING
# =============================================================================
def run_meep_worker(config, params, queue):
    """
    This function runs in a completely separate process.
    It builds the sim, runs it, calculates the score, puts it in the queue, and DIES.
    """
    try:
        # Import inside worker to ensure clean state
        from meep_utils import simulation
        
        # Unpack parameters
        radius, height, n_index = params
        radii, heights, epsilons = [radius], [height], [n_index**2]

        # Build Sim
        sim, dft_obj, flux_obj, _ = simulation.build_sim(
            config, 
            radii=radii, 
            heights=heights, 
            epsilons=epsilons
        )
        
        '''radius, height, n_index, loss = params
        radii, heights, indices, losses = [radius], [height], [n_index], [loss]

        # Build Sim
        sim, dft_obj, flux_obj, _ = simulation.build_sim(
            config, 
            radii=radii, 
            heights=heights, 
            indices=indices,
            losses=losses
        )'''
        
        # Run Sim
        check_pt = mp.Vector3(0, 0, 0)
        sim.run(
            mp.stop_when_fields_decayed(50, mp.Ey, check_pt, 1e-3),
            until=config['simulation']['until']
        )
        #sim.run(until=config['simulation']['until'])
    
        # Calculate Score
        if config['optimization']['target_metric'] == "contrast":
            # (Assuming we are doing the "Contrast Score" logic)
            ex = sim.get_dft_array(dft_obj, mp.Ex, 1)
            ey = sim.get_dft_array(dft_obj, mp.Ey, 1)
            ez = sim.get_dft_array(dft_obj, mp.Ez, 1)
            intensity = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2
            nz = intensity.shape[2]
            center_idx = nz // 2
            #slice_plane = intensity[:, :, center_idx]
            slice_plane = intensity
            
            max_val = np.max(slice_plane)
            mean_val = np.mean(slice_plane)
            
            score = 0.0
            if mean_val != 0:
                score = (max_val / mean_val) - 1.0
        elif config['optimization']['target_metric'] == "goldilocks":
            # 1. Get COMPLEX fields (Do not use abs() yet!)
            # We focus on Ey because the source is Ey-polarized.
            # Index 1 assumes you want the center frequency.
            ey_complex = sim.get_dft_array(dft_obj, mp.Ey, 1)
            
            # 2. Calculate Transmission (Intensity Proxy)
            # We average the intensity over the slice.
            # (Ideally, normalize by input intensity, assuming ~1.0 here for relative scoring)
            intensity_map = np.abs(ey_complex)**2
            mean_transmission = np.mean(intensity_map)
            
            # 3. Calculate Phase
            # We average the complex field *vector* first, then take the angle.
            # This gives the "coherent mean phase" of the wavefront.
            avg_complex_field = np.mean(ey_complex)
            sim_phase = np.angle(avg_complex_field) # Returns value in [-pi, pi]
            
            # 4. The "Goldilocks" Score
            # Target Phase: Pi (3.14159)
            # We use a Cosine metric: 1.0 if phase matches exactly, 0.0 if 90 deg off.
            target_phase = np.pi
            
            # Phase Score: Goes from 0.0 to 1.0
            # cos((sim - target)/2)^2 is a smooth bell curve peaking at target
            phase_alignment = np.cos((sim_phase - target_phase) / 2)**2
            
            # Transmission Score: We want T to be high (e.g., closer to 1.0 is better)
            # We clip it to avoid rewarding super-focusing artifacts
            trans_score = np.clip(mean_transmission, 0, 1.5) / 1.5 
            
            # COMBINED SCORE
            # High Score means: Good Transmission AND Correct Phase
            score = trans_score * phase_alignment
        elif config['optimization']['target_metric'] == "bandpass":
            # flux_obj is a Meep Flux object. 
            # mp.get_fluxes(flux_obj) returns a list of powers for each frequency
            flux_data = mp.get_fluxes(flux_obj)
            
            # Mapping indices based on your config's [1.65, 1.55, 1.30]
            # Meep sorts frequencies Low->High.
            # Lambda: [1.65, 1.55, 1.30] -> Freq: [0.60, 0.64, 0.77]
            # So Index 0 is 1.65um, Index 1 is 1.55um, Index 2 is 1.30um
            
            power_155 = flux_data[1] # Signal
            power_130 = flux_data[2] # Noise
            
            # 3. NORMALIZE (Optional but good practice)
            # Ideally, you run a "normalization run" (no pillar) to get input power.
            # For now, we assume input is roughly constant across small bandwidths.
            
            # 4. SCORING: Log Contrast
            # We want 1.55 to be HIGH and 1.30 to be LOW.
            # A simple ratio is unstable if 1.30 is near zero.
            # Log10 makes the score manageable (e.g., 100x contrast = 2.0).
            
            epsilon = 1e-9 # Prevent divide by zero
            ratio = (power_155 + epsilon) / (power_130 + epsilon)
            
            score = np.log10(ratio)
            
            # Penalize if the "Pass" band is actually dark (Transmission < 10%)
            # (We don't want a filter that blocks everything)
            # Assuming max flux ~ 1.0 (relative)
            if power_155 < 0.1: 
                score -= 5.0 # Heavy penalty
        elif config['optimization']['target_metric'] == "notch-filter":
            # Get Flux at 1.55um (Index 1)
            flux_data = mp.get_fluxes(flux_obj)
            power_155 = flux_data[1] 
            
            # --- THE METRIC ---
            # We want Power -> 0.
            # BO wants to MAXIMIZE score.
            # Score = -1 * Log10(Power)
            # If T=1.0 (Transparent) -> Score = 0
            # If T=0.01 (Blocked)    -> Score = 2
            
            # Add epsilon to avoid log(0)
            score = -1.0 * np.log10(power_155 + 1e-6)
            
        # Send result back to main process
        queue.put(float(score))
        
    except Exception as e:
        queue.put(e) # Send error back if something crashes

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
    def __init__(self, config, sims_dir, animator=None):
        self.config = config
        self.sims_dir = sims_dir
        self.animator = animator
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
    
    def _parameter_mapper_2x(self, x_normalized):
        """
        Maps the optimizer's input vector (0-1 range) to physical units.
        Assumes x is [radius, height, index]
        """
        bounds = self.config['optimization']['parameters']
        
        # Unpack bounds
        r_min, r_max = bounds[0]['bounds']
        h_min, h_max = bounds[1]['bounds']
        n_min, n_max = bounds[2]['bounds']
        k_min, k_max = bounds[3]['bounds']
        
        # De-normalize
        r = x_normalized[0] * (r_max - r_min) + r_min
        h = x_normalized[1] * (h_max - h_min) + h_min
        n = x_normalized[2] * (n_max - n_min) + n_min
        k = x_normalized[3] * (k_max - k_min) + k_min
        
        return float(r), float(h), float(n), float(k)

    '''def run_sim_and_score(self, x_in):
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
        
        if "broadband" in self.config["optimization"]["target_metric"]:
        
            scores = []
            
            # Let's say we have 3 points: 1.65, 1.55, 1.3
            # We iterate through them
            for freq_idx in range(3): 
                
                # Extract fields for this specific frequency
                ex = sim.get_dft_array(dft_obj, mp.Ex, freq_idx)
                ey = sim.get_dft_array(dft_obj, mp.Ey, freq_idx)
                ez = sim.get_dft_array(dft_obj, mp.Ez, freq_idx)
                
                # Calculate scalar contrast for this wavelength
                # (Reusing your existing contrast logic)
                s = self._calculate_contrast_score_raw(ex, ey, ez)
                scores.append(s)
                
            sim.reset_meep()
            del sim
            gc.collect()

            # --- AGGREGATION STRATEGY ---
            
            if self.config["optimization"]["target_metric"] == "broadbandA":
                # Strategy A: Broadband Performance (Average)
                final_score = np.mean(scores)
            elif self.config["optimization"]["target_metric"] == "broadbandB":
                # Strategy B: Filter (Maximize Center, Penalize Sides)
                # Assuming index 1 is center (1.55)
                final_score = scores[1] - 0.5 * (scores[0] + scores[2])
            
            print(f"   [Spectral] 1.65um: {scores[0]:.3f} | 1.55um: {scores[1]:.3f} | 1.3um: {scores[2]:.3f}")
            print(f"   [Result] Aggregated Score: {final_score:.4f}")
            
            return float(final_score)
        
        elif self.config["optimization"]["target_metric"] == "contrast":
            # 4. In-Memory Data Extraction (The Efficiency Hack)
            
            # We only grab the center wavelength for optimization speed
            target_wl = self.config['source']['wavelength']
            # Find index of target_wl in the flux object (assuming center freq)
            freq_idx = 0
            #freq_idx = 1
            
            # Extract fields directly to RAM
            # Shape: [x, y, z] complex128
            ex = sim.get_dft_array(dft_obj, mp.Ex, freq_idx)
            ey = sim.get_dft_array(dft_obj, mp.Ey, freq_idx)
            ez = sim.get_dft_array(dft_obj, mp.Ez, freq_idx)
            
            # Clean up MEEP object to free memory immediately
            sim.reset_meep()
            del sim
            gc.collect()

            # Compute Figure of Merit (Score)
            #score = self._calculate_focusing_score(ex, ey, ez)
            #print(f"Result: Score = {score:.5f}")
            #return score
            
            # Calculate Score (Contrast Ratio)
            return self._calculate_contrast_score(ex, ey, ez, radius, height, n_index)
        
        elif self.config["optimization"]["target_metric"] == "phase-matching":
            ey = sim.get_dft_array(dft_obj, mp.Ey, 1)
            
            # Clean up MEEP object to free memory immediately
            sim.reset_meep()
            del sim
            gc.collect()
            
            return self._score_phase_matching(ey)'''
            
    def run_sim_and_score(self, x_in):
        self.sim_count += 1
        print(f"\n--- Starting Sim #{self.sim_count} (In Isolated Process) ---")
        
        # 1. Map Parameters
        real_params = self._parameter_mapper(x_in)
        print(f"Params: Radius={real_params[0]:.3f}, Height={real_params[1]:.3f}, Index={real_params[2]:.3f}")


        # 2. Setup Multiprocessing
        # We use 'spawn' context to ensure a clean start, compatible with MEEP/MPI
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        
        # 3. Launch the Worker
        p = ctx.Process(target=run_meep_worker, args=(self.config, real_params, queue))
        p.start()
        
        # 4. Wait for Result
        result = queue.get() # Blocks until worker puts data
        p.join() # Clean up the process handle
        
        # 5. Handle Result
        if isinstance(result, Exception):
            print(f"Simulation Failed: {result}")
            return 0.0
            
        print(f"Result: Score = {result:.5f}")
        return result
        
    def _score_phase_matching(self, ey, target_phase_rad=3.14159):
        # 1. Get the complex field averaged over the center (to avoid edge noise)
        # Using Ey assuming source is y-polarized
        complex_field = np.mean(ey) 
        
        # 2. Define target as a complex unit vector
        target_vector = np.exp(1j * target_phase_rad)
        
        # 3. Project simulated field onto target
        # This maximizes BOTH transmission magnitude AND phase alignment.
        # Score is high only if transmission is high AND phase is correct.
        score = np.real(complex_field * np.conjugate(target_vector))
        
        return float(score)

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
        if self.config['plot']:
            fig = plt.figure(figsize=(5, 4))
            plt.imshow(slice_plane, cmap='inferno')
            plt.colorbar(label='Intensity')
            plt.title(f"Sim #{self.sim_count}\nR={r:.2f} H={h:.2f} n={n:.2f}\nScore: {score:.3f}")
            plt.tight_layout()
            #plt.savefig(os.path.join(self.sims_dir, f"sim_{self.sim_count:03d}.png"))
            #plt.close()
            if self.animator:
                self.animator.add_current_figure(fig)
            plt.close(fig)
        
        return float(score)
    
    def _calculate_contrast_score_raw(self, ex, ey, ez):
            """
            Helper method that just does the math (no plotting)
            """
            intensity = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2
            nz = intensity.shape[2]
            center_idx = nz // 2
            slice_plane = intensity[:, :, center_idx]
            
            max_val = np.max(slice_plane)
            mean_val = np.mean(slice_plane)
            
            if mean_val == 0: return 0.0
            return (max_val / mean_val) - 1.0

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
        
        if self.config["plot"] == True:
            self.animator = OptimizationAnimator(self.sims_dir, filename="evolution.mp4", fps=5)
        else:
            self.animator = None
        
        self.oracle = MeepOracle(self.config, self.sims_dir, self.animator)
        self.n_params = len(self.config['optimization']['parameters'])
        self.n_init = self.config['optimization']['n_init']
        self.method = self.config['optimization']['method']
        
        self.train_x = []
        self.train_y = []
        self.history_best = []
        
    def _log_to_csv(self, iteration, score, params, best_so_far):
        # Map normalized params back to real units for readable logs
        real_params = self.oracle._parameter_mapper(params)
        print(f'real_params: {real_params}')
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
            print(f'num_params: {self.n_params}')
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
        
        # initialize the tracker
        tracker = AcquisitionTracker(self.config, self.plots_dir)
        
        for i in range(n_iters):
            iteration_id = len(self.history_best)
            print(f"\n=== Optimization Iteration {i+1}/{n_iters} ===")
            print(f"Current Best Score: {best_value:.5f}")
            
            if self.method == 'bo': # bayesian optimization
                if self.config['optimization']['ard'] == True:
                    ard_setting = self.n_params
                else:
                    ard_setting=None
                # setup kernel
                if self.config['optimization']['kernel'] == 'matern':
                    kernel = MaternKernel(nu=self.config['optimization']['matern_nu'], ard_num_dims=ard_setting)
                elif self.config['optimization']['kernel'] == 'rbf':
                    kernel = RBFKernel(ard_num_dims=ard_setting)

                covar_module = ScaleKernel(kernel)
                
                # 1. Fit Gaussian Process
                gp = SingleTaskGP(
                    train_x, 
                    train_y, 
                    covar_module=covar_module
                )
                #gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                
                # 2. Optimize Acquisition Function (Expected Improvement)
                if self.config['optimization']['acqf_func'] == 'ei':
                    acqf_func = LogExpectedImprovement(gp, best_f=best_value)
                elif self.config['optimization']['acqf_func'] == 'ucb':
                    acqf_func = UpperConfidenceBound(model=gp, beta=self.config['optimization']['ucb_beta'])
                
                # Find the best new point to try
                # bounds are [0,1] because we normalized inside the Oracle
                bounds = torch.stack([
                    torch.zeros(self.n_params, dtype=torch.double), 
                    torch.ones(self.n_params, dtype=torch.double)
                ])
                new_x, _ = optimize_acqf(
                    acqf_func, bounds=bounds, q=1, 
                    num_restarts=self.config["optimization"]["num_restarts"], 
                    raw_samples=self.config["optimization"]["raw_samples"]
                )
                #new_x, _ = optimize_acqf(
                #    EI, bounds=bounds, q=1, num_restarts=40, raw_samples=2048
                #)
                
                best_idx = train_y.argmax()
                best_x_norm = train_x[best_idx]
                tracker.log_step(gp, acqf_func, best_x_norm)
                
                if self.config["plot"]:
                    # VISUALIZE ACQUISITION
                    #self._plot_acquisition_slice(gp, acqf_func, new_x[0], iteration_id)
                    # VISUALIZE ACQUISITION and UNCERTAINTY
                    self._plot_brain_scan(gp, acqf_func, new_x[0], train_x, iteration_id)
                
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
            
            if self.config["plot"]:
                # update Performance Plots
                self._plot_performance_trace()

            # Save Checkpoint
            if i % 5 == 0:
                save_path = os.path.join(self.run_dir, 'checkpoint.pt')
                torch.save({'x': train_x, 'y': train_y}, save_path)
                
        tracker.plot_timeline()

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
        plt.close('all')

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
        plt.close('all')
        
    def _plot_brain_scan(self, gp, acq_func, candidate, train_x, iteration):
        """
        Generates a 3-panel diagnostic plot: GP Mean, GP Uncertainty, and Acquisition Function.
        Takes a slice through the parameter space at the 'Index' of the candidate point.
        """
        import matplotlib.pyplot as plt
        import torch

        # 1. Setup the Slice Grid
        # We will plot Radius (X) vs Height (Y), holding Index (Z) constant at the candidate's value.
        # This assumes the param order is [Radius, Height, Index]
        
        n_grid = 50
        x = torch.linspace(0, 1, n_grid, dtype=torch.double) # Normalized Radius
        y = torch.linspace(0, 1, n_grid, dtype=torch.double) # Normalized Height
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # Fixed Index value (from the candidate)
        fixed_index = candidate[2].item()
        
        # Create the grid of test points [Radius, Height, Fixed_Index]
        # Shape: (2500, 3)
        grid_flat = torch.stack([
            X.flatten(), 
            Y.flatten(), 
            torch.full_like(X.flatten(), fixed_index)
        ], dim=1)

        # 2. Evaluate the Model (GP)
        gp.eval() # Set to evaluation mode
        with torch.no_grad():
            posterior = gp.posterior(grid_flat)
            mean = posterior.mean.squeeze().reshape(n_grid, n_grid).numpy()
            sigma = posterior.variance.sqrt().squeeze().reshape(n_grid, n_grid).numpy()
            
            # Evaluate Acquisition Function
            # Acqf expects input shape (N, 1, d) -> (2500, 1, 3)
            acq_values = acq_func(grid_flat.unsqueeze(1))
            acq = acq_values.reshape(n_grid, n_grid).numpy()

        # 3. Filter Training Points for Overlay
        # We only want to plot previous samples that are "nearby" in terms of Refractive Index
        # otherwise the plot gets cluttered with points that are actually far away in 3D space.
        train_x_np = train_x.numpy()
        # Find points where Index is within +/- 0.1 (normalized) of the slice
        mask = np.abs(train_x_np[:, 2] - fixed_index) < 0.1
        nearby_points = train_x_np[mask]

        # 4. Generate the Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Common plotting kwargs
        extent = [0, 1, 0, 1]
        origin = 'lower'
        
        # --- Panel A: GP Mean (The "Map") ---
        im1 = axes[0].imshow(mean, extent=extent, origin=origin, cmap='viridis')
        axes[0].set_title('GP Posterior Mean (Predicted Score)')
        fig.colorbar(im1, ax=axes[0])
        
        # --- Panel B: GP Uncertainty (The "Fog") ---
        im2 = axes[1].imshow(sigma, extent=extent, origin=origin, cmap='plasma')
        axes[1].set_title('GP Uncertainty (Sigma)')
        fig.colorbar(im2, ax=axes[1])
        
        # --- Panel C: Acquisition Function (The "Decision") ---
        im3 = axes[2].imshow(acq, extent=extent, origin=origin, cmap='inferno')
        axes[2].set_title('Acquisition Function (Exp. Improvement)')
        fig.colorbar(im3, ax=axes[2])

        # 5. Overlays (Context)
        for ax in axes:
            ax.set_xlabel('Normalized Radius')
            ax.set_ylabel('Normalized Height')
            
            # Plot historical points (Black dots)
            if len(nearby_points) > 0:
                ax.scatter(nearby_points[:, 0], nearby_points[:, 1], c='k', s=20, alpha=0.5, label='History')
            
            # Plot the NEW candidate (Red Star)
            ax.scatter(candidate[0], candidate[1], c='r', marker='*', s=200, edgecolors='w', label='Next Sim')

        # Add legend only to the last plot to save space
        axes[2].legend(loc='upper right')
        
        plt.suptitle(f"Brain Scan Iteration {iteration} | Slice at Norm. Index = {fixed_index:.2f}", fontsize=14)
        plt.tight_layout()
        
        # Save
        filename = os.path.join(self.plots_dir, f"brain_scan_iter_{iteration:03d}.png")
        plt.savefig(filename)
        plt.close('all')

if __name__ == "__main__":
    controller = AutonomousController("supplementary/bo_study/config.yaml")
    controller.initialize_data()
    controller.run_optimization_loop()