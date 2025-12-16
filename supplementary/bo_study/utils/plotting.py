import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import glob
import imageio.v2 as imageio
from pathlib import Path
import numpy as np
import pandas as pd

def plot_comparative_performance(csv_paths, labels=None, output_path="comparison_trace.png"):
    """
    Plots the 'Best_So_Far' trace for multiple optimization runs on a single figure.

    Args:
        csv_paths (list): List of file paths to optimization_log.csv files.
        labels (list, optional): Custom names for the legend (e.g., ['Bayesian Opt', 'Random Search']).
                                 If None, defaults to the run folder name.
        output_path (str): Where to save the resulting image.
    """
    plt.figure(figsize=(10, 6))
    
    # Use a high-contrast color cycle
    colors = plt.cm.tab10.colors 

    for i, path in enumerate(csv_paths):
        iterations = []
        best_scores = []

        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue

        # Read the CSV
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    iterations.append(int(row['Iteration']))
                    best_scores.append(float(row['Best_So_Far']))
                except (ValueError, KeyError):
                    continue # Skip empty or malformed lines

        # Determine Legend Label
        if labels and i < len(labels):
            lbl = labels[i]
        else:
            # Fallback: Use the parent folder name (e.g., "run_20251212_1400")
            # Assumes structure: .../run_ID/optimization_log.csv
            lbl = os.path.basename(os.path.dirname(path))

        # Plot Trace
        color = colors[i % len(colors)]
        plt.plot(iterations, best_scores, linewidth=2.5, label=lbl, color=color)
        
        # Add a marker at the final point for clarity
        if iterations:
            plt.plot(iterations[-1], best_scores[-1], 'o', color=color, markersize=6)

    # Styling
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Score Achieved", fontsize=12)
    plt.title("Optimization Strategy Comparison", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.minorticks_on()
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot generated successfully: {output_path}")
    
def animate_existing_visuals(sim_visuals_dir, output_name="evolution.gif", fps=4, type="sim"):
    """
    Reads existing sim_XXX.png files from a directory and creates a video/gif.

    Args:
        sim_visuals_dir (str): Path to the folder containing png files.
        output_name (str): Filename for output (e.g., 'evolution.gif' or 'evolution.mp4').
                           The extension determines the format.
        fps (int): Frames per second.
    """
    # 1. Find and Sort Files
    if type == "sim":
        # Since filenames are zero-padded (sim_001, sim_002), standard sort works correctly.
        search_pattern = os.path.join(sim_visuals_dir, "sim_*.png")
    elif type == "acq":
        search_pattern = os.path.join(sim_visuals_dir, "acq_func_iter_*.png")

    filenames = sorted(glob.glob(search_pattern))

    if not filenames:
        print(f"Error: No '.png' files found in {sim_visuals_dir}")
        return

    print(f"Found {len(filenames)} images in {sim_visuals_dir}. Preparing animation...")

    # 2. Read Images into Memory
    images = []
    for filename in filenames:
        # imageio.imread loads the image as a numpy array
        images.append(imageio.imread(filename))

    # 3. Write the Animation
    output_path = os.path.join(sim_visuals_dir, output_name)
    
    # Extra arguments for GIF saving to ensure it loops and optimizes size
    kwargs = {}
    if output_name.lower().endswith('.gif'):
        kwargs = {'loop': 0, 'optimize': True} # loop=0 means infinite loop

    print(f"Writing animation to {output_path}...")
    # imageio detects format from file extension (GIF vs MP4)
    imageio.mimsave(output_path, images, fps=fps, **kwargs)
    print("Animation complete.")
    
class OptimizationAnimator:
    """
    Helper class to incrementally build a video file frame by frame 
    from Matplotlib figures during an optimization loop.
    """
    def __init__(self, output_dir, filename="evolution.mp4", fps=4):
        self.filepath = os.path.join(output_dir, filename)
        self.frame_count = 0
        
        # Initialize the video writer.
        # 'FFMPEG' backend is robust for MP4s. 
        # Use format='GIF' and mode='I' if you prefer GIFs, but MP4 is better for streaming writes.
        self.writer = imageio.get_writer(self.filepath, fps=fps, format='FFMPEG', mode='I', quality=9)
        print(f"[Animator] Initialized video writer: {self.filepath}")

    def add_current_figure(self, fig):
        """
        Takes a matplotlib figure object, converts it to an RGB image array 
        in memory, and appends it to the video stream.
        """
        # 1. Force Matplotlib to draw the canvas so data exists
        fig.canvas.draw()

        # 2. Extract the raw RGB buffer from the canvas
        # (This avoids saving to disk temporarily)
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        
        # Reshape buffer into an image array (Height, Width, 3 Channels)
        image_array = buf.reshape((h, w, 3))

        # 3. Append to video
        self.writer.append_data(image_array)
        self.frame_count += 1
        # Optional: print dot to show activity without clutter
        # print(".", end="", flush=True) 

    def close(self):
        """Finalizes the video file. Must be called at end of run."""
        self.writer.close()
        print(f"\n[Animator] Video finalized. Total frames: {self.frame_count}")
        
def plot_aggregated_convergence(bo_paths, rs_paths):
    plt.figure(figsize=(10, 6))

    for name, paths, color in [("Bayesian Opt", bo_paths, 'blue'), ("Random Search", rs_paths, 'gray')]:
        all_series = []
        
        # 1. Load Data
        for p in paths:
            df = pd.read_csv(p)
            # Ensure we track the 'best so far' cumulatively
            # (Your CSV already has 'Best_So_Far', but doing it manually ensures safety)
            best_trace = df['Score'].cummax()
            all_series.append(best_trace.values)
        
        # 2. Compute Statistics
        # Pad arrays with NaN if lengths differ, or truncate to min length
        min_len = min(len(s) for s in all_series)
        data_matrix = np.array([s[:min_len] for s in all_series])
        
        mean_trace = np.mean(data_matrix, axis=0)
        std_trace = np.std(data_matrix, axis=0)
        iters = np.arange(min_len)

        # 3. Plot Mean
        plt.plot(iters, mean_trace, label=name, color=color, linewidth=3)
        
        # 4. Plot Confidence Interval (Mean +/- 1 Std Dev)
        plt.fill_between(iters, mean_trace - std_trace, mean_trace + std_trace, 
                         color=color, alpha=0.2)

    plt.xlabel("Simulation Count", fontsize=12)
    plt.ylabel("Best Contrast Score", fontsize=12)
    plt.title("Statistical Comparison: BO vs Random Search (N=5 Trials)", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig("statistical_comparison.png")

def plot_param_search_space(csv_file_path, output_dir=None):
    """
    Generates 2D scatter plots of parameter pairs from an optimization log.
    Points are colored by Score to visualize high-performing regions.
    
    Args:
        csv_file_path (str): Path to optimization_log.csv
        output_dir (str): Folder to save plots. If None, saves in the same folder as csv.
    """
    
    # 1. Setup paths
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
        
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found: {csv_file_path}")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Filter for the relevant columns
    # Adjust column names if your CSV header differs (e.g. 'n' vs 'Index')
    params = ['Radius', 'Height', 'Index']
    score_col = 'Score'
    
    # Check if columns exist
    if not all(col in df.columns for col in params):
        print(f"Error: CSV is missing one of the parameter columns: {params}")
        return

    # 3. Create the Plot Grid (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # Define pairs to plot
    pairs = [
        ('Radius', 'Height'),
        ('Radius', 'Index'),
        ('Height', 'Index')
    ]
    
    # Find the global best point to highlight it
    best_idx = df[score_col].idxmax()
    best_row = df.loc[best_idx]

    # 4. Generate Scatter Plots
    for ax, (x_name, y_name) in zip(axes, pairs):
        
        # Scatter plot: Color by Score (cmap='viridis' or 'plasma')
        # We sort by score so high-scoring points are plotted ON TOP of low-scoring ones
        df_sorted = df.sort_values(by=score_col)
        
        sc = ax.scatter(
            df_sorted[x_name], 
            df_sorted[y_name], 
            c=df_sorted[score_col], 
            cmap='viridis', 
            s=50, 
            alpha=0.8,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Highlight the Best Point found
        ax.scatter(
            best_row[x_name], 
            best_row[y_name], 
            c='red', 
            s=200, 
            marker='*', 
            edgecolors='white',
            linewidth=1.5,
            label='Best Found'
        )
        
        ax.set_xlabel(x_name, fontsize=12)
        ax.set_ylabel(y_name, fontsize=12)
        ax.set_title(f"{x_name} vs. {y_name}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add a localized legend for the star
        if ax == axes[0]:
            ax.legend(loc='upper left', frameon=True)

    # Add a global colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Contrast Score', fontsize=12)

    # 5. Save
    run_name = os.path.basename(os.path.dirname(csv_file_path))
    save_path = os.path.join(output_dir, f"search_scatter_{run_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Search scatter plot saved to: {save_path}")
    
def plot_rs_histogram(csv_file_path, output_dir=None):
    
    # 1. Setup paths
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
        
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found: {csv_file_path}")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
    
    # 3. Plottttt
    scores = pd.to_numeric(df['Score'], errors='coerce').dropna()
    plt.hist(scores, bins=10)
    plt.xlabel('Contrast Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contrast Evaluations')

    # 4. Save
    run_name = os.path.basename(os.path.dirname(csv_file_path))
    save_path = os.path.join(output_dir, "score_histogram.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def plot_3d_resonance_cloud(csv_path):
    # 1. Load Data
    if not os.path.exists(csv_path):
        print("CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 2. Extract Columns (Ensure names match your CSV)
    r = df['Radius']
    h = df['Height']
    n = df['Index']
    score = df['Score']

    # 3. Create 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 4. Scatter Plot
    # - C (Color) maps to Score
    # - S (Size) maps to Score (makes good points bigger/easier to see)
    # - Alpha (Transparency) helps see inside the cloud
    
    # Normalize size: Smallest dot 10, Biggest dot 200
    sizes = 10 + (score - score.min()) / (score.max() - score.min()) * 200
    
    img = ax.scatter(r, h, n, c=score, s=sizes, cmap='inferno', alpha=0.8, edgecolors='k', linewidth=0.2)

    # 5. Labels and Aesthetics
    ax.set_xlabel('Radius (microns)', fontsize=12, labelpad=10)
    ax.set_ylabel('Height (microns)', fontsize=12, labelpad=10)
    ax.set_zlabel('Refractive Index', fontsize=12, labelpad=10)
    ax.set_title(f'Optimization Landscape (N={len(df)} Points)', fontsize=16)

    # Add Colorbar
    cbar = fig.colorbar(img, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Contrast Score', fontsize=12)

    # Highlight the Best Point
    best_idx = score.idxmax()
    ax.scatter(r[best_idx], h[best_idx], n[best_idx], c='red', s=300, marker='*', label='Global Max')
    ax.legend()

    # Set initial viewing angle (Elev=30, Azim=45 is standard isometric)
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig('3d_resonance_cloud.png', dpi=150)
    print("Plot saved to 3d_resonance_cloud.png")

if __name__ == "__main__":
    
    files_to_compare = [
        "../../results/run_20251215_210258/optimization_log.csv",
        "../../results/run_20251215_223117/optimization_log.csv",
        "../../results/run_20251215_224426/optimization_log.csv",
        "../../results/run_20251215_225215/optimization_log.csv",
        "../../results/run_20251215_234303/optimization_log.csv",
        "../../results/run_20251212_205037/optimization_log.csv",
        "../../results/run_20251212_214348/optimization_log.csv",
    ]
    
    custom_labels = [
        "Bayesian Optimization (Matern 1.5, UCB)",
        "Bayesian Optimization (Matern 2.5, UCB)",
        "Bayesian Optimization (Matern 2.5, EI)",
        "Bayesian Optimization (RBF, EI)",
        "Bayesian Optimization (No ARD)",
        "Random Search",
        "Quasi-Random Search (Sobol)"
    ]
    
    plot_comparative_performance(files_to_compare, labels=custom_labels)
    
    bo_runs = [
        "../../results/run_20251212_194120/optimization_log.csv",
        "../../results/run_20251215_200314/optimization_log.csv",
        "../../results/run_20251215_202656/optimization_log.csv",
        "../../results/run_20251215_210258/optimization_log.csv",
        "../../results/run_20251215_210907/optimization_log.csv",
    ]
    
    bo_runs_matern15 = [
        "../../results/run_20251215_170004/optimization_log.csv",
        "../../results/run_20251215_213817/optimization_log.csv",
    ]
    
    rs_runs = [
        "../../results/run_20251212_205037/optimization_log.csv",
        "../../results/run_20251212_214348/optimization_log.csv",
        "../../results/run_20251215_211523/optimization_log.csv",
        "../../results/run_20251215_212038/optimization_log.csv",
    ]
    
    #plot_param_search_space(files_to_compare[2])
    
    #plot_aggregated_convergence(bo_runs_matern15, rs_runs)
    
    #plot_rs_histogram(rs_runs[2])
    
    #plot_3d_resonance_cloud(bo_runs[0])
    
    #sim_dir = str(os.path.join(str(Path(files_to_compare[0]).parent), "plots"))
    #animate_existing_visuals(sim_dir, output_name="evolution_flipbook.gif", fps=10, type='acq')