import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_accuracy_vs_density(results_dir, output_plot_path):
    """
    Generates a scatter plot showing accuracy (relative density) vs. network density.

    Args:
        results_dir (str): Directory containing CSV files with experiment results.
        output_plot_path (str): Path to save the generated plot.
    """

    all_data = []

    # 1. Verify results_directory (use absolute path if possible)
    results_dir = os.path.abspath(results_dir)
    print(f"Results directory: {results_dir}")

    # 2. Check for CSV files
    csv_files = [filename for filename in os.listdir(results_dir) if filename.endswith(".csv")]
    if not csv_files:
        print(f"Error: No CSV files found in {results_dir}")
        return

    for filename in csv_files: # Iterate through the filtered list
        file_path = os.path.join(results_dir, filename)
        print(f"Reading file: {file_path}") # 3. Inspect file path

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {filename} is empty. Skipping.")
            continue  # Skip to the next file
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}. Skipping.")
            continue

        dataset_name = os.path.splitext(filename)[0].replace("_results", "")
        avg_initial_density = df['Initial Density'].mean()

        grouped_data = df.groupby('Epsilon').agg({
            'LDP Density': 'mean',
            'Baseline Density': 'mean'
        }).reset_index()

        grouped_data['Accuracy'] = grouped_data['LDP Density'] / grouped_data['Baseline Density']
        grouped_data['Dataset'] = dataset_name
        grouped_data['Initial Density'] = avg_initial_density

        # 4. Handle empty DataFrames (less likely, but good practice)
        if not grouped_data.empty:
            all_data.append(grouped_data)
        else:
            print(f"Warning: {filename} resulted in an empty DataFrame.")

    if not all_data: # Check if all_data is still empty after processing
        print("Error: No data to concatenate after processing all files.")
        return

    final_df = pd.concat(all_data)

    # Plotting code (same as before)
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{txfonts}',
        'pgf.rcfonts': False
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    epsilons = final_df['Epsilon'].unique()
    colors = ['red', 'green', 'blue']

    for i, eps in enumerate(epsilons):
        data_subset = final_df[final_df['Epsilon'] == eps]
        ax.scatter(data_subset['Initial Density'], data_subset['Accuracy'], color=colors[i], label=f'Epsilon = {eps}', alpha=0.7)

        z = np.polyfit(data_subset['Initial Density'], data_subset['Accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(data_subset['Initial Density'], p(data_subset['Initial Density']), colors[i], linestyle="--", linewidth=0.5)

    ax.set_xlabel('Initial Density', fontsize=14)
    ax.set_ylabel('Accuracy (LDP Density / Baseline Density)', fontsize=14)
    ax.set_title('Accuracy vs. Initial Density', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='-', alpha=0.5)

    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example Usage:
results_directory = "txt12" # Replace with the directory where your CSV files are saved
output_plot_path = "accuracy_vs_initial_density.png"
plot_accuracy_vs_density(results_directory, output_plot_path)