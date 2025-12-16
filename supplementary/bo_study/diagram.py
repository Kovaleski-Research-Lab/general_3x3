import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bo_workflow_diagram(filename="bo_workflow_diagram.png"):
    """
    Generates a schematic diagram of the specific Bayesian Optimization 
    loop used for the Meep Notch Filter optimization.
    """
    # Create Figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off') # Hide axes

    # --- STYLE CONFIG ---
    box_props = dict(boxstyle="round,pad=0.5", ec="black", lw=2)
    font_title = {'weight': 'bold', 'size': 12}
    font_desc = {'size': 10, 'style': 'italic'}
    
    # --- HELPER TO DRAW ARROWS ---
    def draw_arrow(start, end, label=None):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=2, color="#444444"))
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                    fontsize=9, color="#666666")

    # --- 1. DATA / HISTORY (Top Center) ---
    ax.text(5, 9, "START / HISTORY", ha="center", va="center", weight="bold")
    rect_hist = patches.FancyBboxPatch((3.5, 8.2), 3, 1.2, boxstyle="round,pad=0.2", 
                                       fc="#E6E6FA", ec="black", lw=2)
    ax.add_patch(rect_hist)
    ax.text(5, 8.8, "Dataset (X, Y)", ha="center", va="center", **font_title)
    ax.text(5, 8.5, "X: [Radius, Height, Index]\nY: Score", ha="center", va="center", **font_desc)

    # --- 2. SURROGATE MODEL (Right Top) ---
    rect_gp = patches.FancyBboxPatch((7.5, 6), 2.2, 1.5, **box_props, fc="#FFFACD")
    ax.add_patch(rect_gp)
    ax.text(8.6, 6.9, "Fit Surrogate Model", ha="center", va="center", **font_title)
    ax.text(8.6, 6.5, "Gaussian Process\nKernel: Matern 1.5/2.5 or RBF", ha="center", va="center", **font_desc)

    # --- 3. ACQUISITION (Right Bottom) ---
    rect_acq = patches.FancyBboxPatch((7.5, 2.5), 2.2, 1.5, **box_props, fc="#FFE4E1")
    ax.add_patch(rect_acq)
    ax.text(8.6, 3.4, "Select Candidate", ha="center", va="center", **font_title)
    ax.text(8.6, 3.0, "Optimize UCB / LogEI\nFind x_next", ha="center", va="center", **font_desc)

    # --- 4. THE ORACLE / MEEP (Bottom Center) ---
    # Draw a larger container for the simulation steps
    rect_oracle = patches.FancyBboxPatch((3.0, 0.5), 4.0, 2.5, boxstyle="round,pad=0.2", 
                                         fc="#E0FFFF", ec="blue", lw=2, linestyle="--")
    ax.add_patch(rect_oracle)
    ax.text(5, 2.7, "THE ORACLE (and MEEP Sim)", ha="center", va="center", weight="bold", color="blue")
    
    # Internal Steps of Oracle
    ax.text(5, 2.2, "1. Build Geometry (Cylinder + Substrate)", ha="center", va="center", size=9)
    ax.text(5, 1.8, "2. Run FDTD (Stop when decayed)", ha="center", va="center", size=9)
    ax.text(5, 1.4, "3. Extract Flux @ 1.55 µm", ha="center", va="center", size=9)
    ax.text(5, 1.0, "4. Calculate Metric: -log(T)", ha="center", va="center", size=9, weight="bold")

    # --- 5. UPDATE (Left) ---
    rect_update = patches.FancyBboxPatch((0.5, 4.5), 2.0, 1.2, **box_props, fc="#F0FFF0")
    ax.add_patch(rect_update)
    ax.text(1.5, 5.3, "Update", ha="center", va="center", **font_title)
    ax.text(1.5, 4.9, "Append (x, y)\nto Dataset", ha="center", va="center", **font_desc)

    # --- CONNECTING ARROWS ---
    
    # History -> GP
    draw_arrow((6.5, 8.2), (8.6, 7.5), "Train")
    
    # GP -> Acq
    draw_arrow((8.6, 6.0), (8.6, 4.0), "Predict Mean/Var")
    
    # Acq -> Oracle
    draw_arrow((7.5, 3.25), (7.0, 1.75), "Suggest x = [r, h, n]")
    
    # Oracle -> Update
    draw_arrow((3.0, 1.75), (2.5, 4.5), "Return Score y")
    
    # Update -> History
    draw_arrow((1.5, 5.7), (3.5, 8.2), "Loop")

    plt.title("Bayesian Optimization Workflow: Metasurface Notch Filter", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Workflow diagram saved to {filename}")
    plt.show()

# Run it
if __name__ == "__main__":
    plot_bo_workflow_diagram()