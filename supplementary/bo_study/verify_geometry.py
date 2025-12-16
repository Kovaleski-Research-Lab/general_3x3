import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Import your geometry logic
# (Ensure your project folder structure allows this import)
from meep_utils.geometries import get_substrate_params 

def plot_z_stack(config):
    # 1. PARAMETER CALCULATION
    params = config.copy()
    
    # Run geometry logic
    params = get_substrate_params(params)
    geo = params['geometry']
    sub = params['substrate_params']
    
    # --- GEOMETRY EXTRACTION ---
    # Cell bounds
    z_cell = params['cell_z']
    z_min_cell = -z_cell / 2
    z_max_cell = z_cell / 2
    
    # PML Thickness
    t_pml = geo['thickness_pml']
    
    # Silica (Substrate)
    h_silica = sub['size_z_fused_silica']
    c_silica = sub['loc_z_fused_silica']
    z_min_silica = c_silica - h_silica/2
    z_max_silica = c_silica + h_silica/2
    
    # PDMS (Superstrate)
    h_pdms = sub['size_z_pdms']
    c_pdms = sub['loc_z_pdms']
    z_min_pdms = c_pdms - h_pdms/2
    z_max_pdms = c_pdms + h_pdms/2
    
    # Pillar
    h_pillar = geo['height_pillar']
    c_pillar = geo['loc_z_pillar']
    z_min_pillar = c_pillar - h_pillar/2
    z_max_pillar = c_pillar + h_pillar/2
    
    # --- MONITOR & SOURCE CALCULATION ---
    # Replicating logic from your simulation setup
    
    # Source Position (Gaussian Source)
    # Logic from sources.py: thickness_pml + ((size_z_fused_silica-thickness_pml) * 0.2) - params['cell_z'] / 2
    z_source = t_pml + ((sub['size_z_fused_silica'] - t_pml) * 0.2) - (z_cell / 2)
    
    # Monitor Position (DFT Plane)
    # Logic from field_monitors.py: loc_top_fused_silica + height_pillar + 0.775
    # Note: loc_top_fused_silica is simply z_max_silica
    z_monitor = z_max_silica + h_pillar + 0.775

    # 2. PLOTTING
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # A. Draw PML Regions (The "Danger Zones")
    # Bottom PML
    rect_pml_bot = patches.Rectangle((-2, z_min_cell), 4, t_pml, 
                                     linewidth=0, facecolor='grey', alpha=0.3, hatch='///', label='PML (Absorber)')
    ax.add_patch(rect_pml_bot)
    # Top PML
    rect_pml_top = patches.Rectangle((-2, z_max_cell - t_pml), 4, t_pml, 
                                     linewidth=0, facecolor='grey', alpha=0.3, hatch='///')
    ax.add_patch(rect_pml_top)

    # B. Draw Geometry
    # Silica (Blue)
    rect_silica = patches.Rectangle((-1, z_min_silica), 2, h_silica, 
                                    linewidth=1, edgecolor='b', facecolor='blue', alpha=0.2, label='Silica Substrate')
    ax.add_patch(rect_silica)
    
    # PDMS (Green)
    rect_pdms = patches.Rectangle((-1, z_min_pdms), 2, h_pdms, 
                                  linewidth=1, edgecolor='g', facecolor='green', alpha=0.2, label='PDMS Superstrate')
    ax.add_patch(rect_pdms)
    
    # Pillar (Red)
    rect_pillar = patches.Rectangle((-0.2, z_min_pillar), 0.4, h_pillar, 
                                    linewidth=1, edgecolor='r', facecolor='red', alpha=0.8, label='Pillar')
    ax.add_patch(rect_pillar)
    
    # C. Draw Monitors & Sources
    # Source Plane
    ax.axhline(z_source, color='cyan', linewidth=2, linestyle='-', label='Source Plane (Gaussian)')
    ax.text(-1.8, z_source + 0.05, "SOURCE", color='cyan', fontweight='bold', fontsize=10)
    
    # Monitor Plane (The Bandpass Measurement)
    ax.axhline(z_monitor, color='magenta', linewidth=2, linestyle='-', label='DFT Monitor Plane (Data)')
    ax.text(-1.8, z_monitor + 0.05, "DATA MONITOR", color='magenta', fontweight='bold', fontsize=10)

    # D. Annotations
    ax.text(1.1, z_max_silica, f"Silica Top:\n{z_max_silica:.4f}", verticalalignment='center', fontsize=8)
    ax.text(1.1, z_min_pillar, f"Pillar Bot:\n{z_min_pillar:.4f}", verticalalignment='center', fontsize=8)
    ax.text(1.1, z_max_pillar, f"Pillar Top:\n{z_max_pillar:.4f}", verticalalignment='center', fontsize=8)
    ax.text(1.1, z_monitor, f"Mon Z:\n{z_monitor:.4f}", verticalalignment='center', fontsize=8, color='magenta')

    # E. Validity Checks
    gap_error = z_min_pillar - z_max_silica
    
    # Check 1: Is Monitor inside PML?
    mon_in_pml = z_monitor > (z_max_cell - t_pml)
    mon_status = "FAIL (Inside PML!)" if mon_in_pml else "OK"
    mon_color = "red" if mon_in_pml else "green"

    # Check 2: Is Source inside PML?
    src_in_pml = z_source < (z_min_cell + t_pml)
    src_status = "FAIL (Inside PML!)" if src_in_pml else "OK"
    
    title_text = (f"Simulation Setup Verification\n"
                  f"Gap Error: {gap_error:.2e} um\n"
                  f"Monitor Status: {mon_status}\n"
                  f"Source Status: {src_status}")
    
    ax.set_title(title_text, color=mon_color if mon_in_pml else 'black')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(z_min_cell - 0.2, z_max_cell + 0.2)
    ax.set_ylabel("Z Position (um)")
    
    # Place legend outside to save space
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("z_stack_check_full.png")
    print(f"Plot saved to z_stack_check_full.png")
    print(f"Monitor Z: {z_monitor}")
    print(f"PML Boundary: {z_max_cell - t_pml}")

# --- MOCK CONFIG ---
mock_config = {
    'grid_size': 1,
    'geometry': {
        'material_index_fused_silica': 1.44,
        'size_z_fused_silica': 0.78,
        'substrate_buffer': False,
        'thickness_pml': 0.78,
        'unit_cell_size': 0.68,
        'material_index_pdms': 1.4,
        'size_z_pdms': 1.56,
        'size_z_fused_silica': 0.78,
        'neighborhood_size': [1,1],
        'atom_type': 'cylinder',
        'height_pillar': 1.052,
        'size_x_buffer': 0, 'size_y_buffer': 0, 'size_z_buffer': 0.75,
        'loc_z_fused_silica': 0, 
        'loc_z_pdms': 0, 
        'loc_z_pillar': 0
    },
    'simulation': {'resolution': 60}
}

if __name__ == "__main__":
    plot_z_stack(mock_config)