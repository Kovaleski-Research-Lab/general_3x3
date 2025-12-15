import matplotlib.pyplot as plt
import csv
import os

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

if __name__ == "__main__":
    
    files_to_compare = [
        "../../results/run_20251212_194120/optimization_log.csv",
        "../../results/run_20251212_205037/optimization_log.csv",
    ]
    
    custom_labels = [
        "Bayesian Optimization (Matern 5/2)",
        "Random Search (Baseline)"
    ]
    
    plot_comparative_performance(files_to_compare, labels=custom_labels)