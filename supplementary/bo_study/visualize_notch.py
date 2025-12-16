import matplotlib.pyplot as plt
import numpy as np
import meep as mp
from meep_utils import simulation
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def visualize_notch_filter_performance(config, params):
    """
    Runs a single simulation for [radius, height, index] and visualizes the physics.
    
    Args:
        config (dict): Your existing configuration dictionary.
        params (list): [radius, height, index] to test.
    """
    print(f"--- Running Visualizer for R={params[0]}, H={params[1]}, n={params[2]} ---")
    
    # 1. SETUP & RUN
    radius, height, n_index = params
    radii, heights, epsilons = [radius], [height], [n_index**2]

    # Re-enable the 2D DFT monitor if it isn't already in your build_sim
    # We need 'flux_obj' for the spectrum and 'dft_obj' for the field picture
    sim, dft_obj, flux_obj, _ = simulation.build_sim(
        config, radii=radii, heights=heights, epsilons=epsilons
    )
    
    # Run longer for visualization to ensure resonance is clear
    # We use a simple run check here
    check_pt = mp.Vector3(0, 0, 0)
    sim.run(
        mp.stop_when_fields_decayed(50, mp.Ey, check_pt, 1e-5),
        until=300 # Generous time budget for high-Q visualization
    )
    
    # 2. EXTRACT SPECTRUM (Flux)
    # Get flux at all frequencies defined in your config
    flux_freqs = mp.get_flux_freqs(flux_obj)
    flux_data = mp.get_fluxes(flux_obj)
    
    # Convert to Wavelength (um) and Sort (MEEP uses Freq, so lists are reversed)
    wavelengths = [1/f for f in flux_freqs]
    transmission = np.array(flux_data)
    
    # 3. EXTRACT FIELDS (Spatial)
    # Get the Ey field at 1.55um (Index 1 usually, but let's be safe)
    # Find index closest to 1.55
    target_wl = 1.55
    #idx_155 = np.argmin(np.abs(np.array(wavelengths) - target_wl))
    idx_155 = 1
    
    ey_data = sim.get_dft_array(dft_obj, mp.Ey, idx_155)
    field_intensity = np.abs(ey_data)**2
    
    # 4. CALCULATE SCORE
    power_155 = transmission[idx_155]
    score = -1.0 * np.log10(power_155 + 1e-6)
    
    # 5. PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PLOT 1: The Spectrum (Color Response) ---
    ax_spec = axes[0]
    ax_spec.plot(wavelengths, transmission, 'b-o', linewidth=2, label='Transmission')
    
    # Highlight the Target (1.55)
    ax_spec.axvline(1.55, color='r', linestyle='--', label='Target (1.55um)')
    ax_spec.plot(wavelengths[idx_155], power_155, 'rx', markersize=12, markeredgewidth=3)
    
    ax_spec.set_xlabel("Wavelength (um)")
    ax_spec.set_ylabel("Transmission (Power)")
    ax_spec.set_title(f"Spectral Response\nScore: {score:.4f} (T={power_155:.2%})")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend()
    ax_spec.set_ylim(-0.05, 1.05)
    
    # --- PLOT 2: The Fields (Physical Mechanism) ---
    ax_field = axes[1]
    # Rotate for better view (Z on Y-axis usually)
    im = ax_field.imshow(field_intensity.T, cmap='inferno', origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax_field, label='|Ey|^2 Intensity')
    
    ax_field.set_title(f"Field Profile at 1.55um\n(R={radius}, H={height}, n={n_index})")
    ax_field.axis('off') # Hide pixels, just show the mode
    
    plt.tight_layout()
    filename = f"viz_R{radius:.3f}_H{height:.2f}_n{n_index:.2f}.png"
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.show()
    
def visualize_notch_normalized(config, params):
    """
    Runs Calibration (Empty) -> Runs Simulation (Device) -> Plots T = Device/Empty
    """
    # 1. PREPARE EMPTY LISTS
    # We need to match the neighborhood size (Nx * Ny) defined in config
    nx, ny = config['geometry']['neighborhood_size']
    num_pillars = nx * ny
    
    # Create lists of ZEROS. 
    # Radius=0 makes the pillar invisible (Calibration).
    radii_empty = [0.0] * num_pillars
    # Height doesn't strictly matter if radius is 0, but we provide valid floats to prevent errors
    heights_empty = [1.0] * num_pillars 
    # Epsilons list can just be dummy values
    epsilons_empty = [1.0] * num_pillars

    print("--- 1. Running CALIBRATION (Empty Sim) ---")
    
    sim_empty, _, flux_empty_obj, _ = simulation.build_sim(
        config, 
        radii=radii_empty, 
        heights=heights_empty, 
        epsilons=epsilons_empty
    )
    
    # Run Empty (Fast run)
    check_pt = mp.Vector3(0, 0, 0)
    sim_empty.run(until=200) 
    
    flux_freqs = mp.get_flux_freqs(flux_empty_obj)
    flux_empty_data = np.array(mp.get_fluxes(flux_empty_obj))
    
    # ---------------------------------------------------------
    
    print(f"--- 2. Running DEVICE (R={params[0]}, H={params[1]}) ---")
    radius, height, n_index = params
    
    # Prepare lists for the actual device
    radii_device = [radius] * num_pillars
    heights_device = [height] * num_pillars
    epsilons_device = [n_index**2] * num_pillars
    
    sim_dev, dft_dev, flux_dev_obj, _ = simulation.build_sim(
        config, 
        radii=radii_device, 
        heights=heights_device, 
        epsilons=epsilons_device
    )
    
    # Run Device (Longer for resonance)
    sim_dev.run(
        mp.stop_when_fields_decayed(50, mp.Ey, check_pt, 1e-5),
        until=400 # 400 is safer for very high-Q notches
    )
    
    flux_dev_data = np.array(mp.get_fluxes(flux_dev_obj))
    
    # --- 3. NORMALIZE & PLOT ---
    
    # Transmission = Flux_Device / Flux_Empty
    # We add 1e-9 to denominator to avoid DivideByZero if source is 0
    transmission = flux_dev_data / (flux_empty_data + 1e-9)
    
    wavelengths = [1/f for f in flux_freqs]
    
    # FIELD PROFILE --------------
    
    ey_empty = sim_empty.get_dft_array(dft_dev, mp.Ey, 1) # Reuse dft object logic
    source_intensity_max = np.max(np.abs(ey_empty)**2)
    
    # 2. GET DEVICE INTENSITY
    ey_dev = sim_dev.get_dft_array(dft_dev, mp.Ey, 1)
    field_intensity = np.abs(ey_dev)**2
    
    # 3. NORMALIZE (Enhancement Factor)
    # This tells us: "How many times brighter is this pixel than the input laser?"
    enhancement_map = field_intensity / source_intensity_max
    
    # 4. PLOT
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use 'inferno' or 'magma' for high contrast
    im = ax.imshow(enhancement_map.T, cmap='magma', origin='lower', aspect='auto')
    
    cbar = fig.colorbar(im, ax=ax, label='Enhancement Factor (|E|² / |E_inc|²)')
    
    ax.set_title(f"Field Enhancement at 1.55um\nMax Enhancement: {np.max(enhancement_map):.1f}x")
    ax.axis('off')
    
    plt.tight_layout()
    filename = f"prof_R{radius:.3f}_h{height:.2f}_n{n_index:.2f}.png"
    plt.savefig(filename)
    plt.close()
    
    # TRANSMISSION SPEC ----------
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot normalized transmission
    ax.plot(wavelengths, transmission, 'b-o', linewidth=2, label='Normalized Transmission')
    
    # Highlight the 1.55um target
    target_idx = np.argmin(np.abs(np.array(wavelengths) - 1.55))
    target_val = transmission[target_idx]
    ax.plot(wavelengths[target_idx], target_val, 'rx', markersize=12, markeredgewidth=3, label=f"T(1.55) = {target_val:.2%}")
    
    # Formatting
    ax.set_ylim(-0.05, 1.1) # Show slightly above 1.0 just in case of numerical noise
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Transmission (Normalized)")
    ax.set_title(f"Spectral Response (Normalized)\nR={radius:.4f}, H={height:.2f}, n={n_index:.2f}")
    
    ax.axvline(1.55, color='r', linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    filename = f"spec_R{radius:.3f}_h{height:.2f}_n{n_index:.2f}.png"
    plt.savefig(filename)
    print(f"Saved normalized plot to {filename}")
    plt.show()

if __name__=="__main__":
    conf = load_config('supplementary/bo_study/config.yaml')
    params_good = [0.21544961119174727,0.7958482187145944,3.4399480326464342]
    params_bad = [0.22987157739698888,1.1710246354341507,3.48]
    #visualize_notch_filter_performance(conf, params_good)
    #visualize_notch_filter_performance(conf, params_bad)
    visualize_notch_normalized(conf, params_good)
# Example Usage:
# Run on the "Best" point BO found
# visualize_notch_filter_performance(config, [0.3619, 1.25, 3.51])
# Run on the "Ideal" point you suspect
# visualize_notch_filter_performance(config, [0.2184, 1.25, 3.48])