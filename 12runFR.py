import os
import time
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from measurements import compare_subgraph_similarity
import pandas as pd
import numpy as np # Ensure numpy is imported for np.number
import matplotlib.pyplot as plt
from our_method import *
from ReadGraph import *
from Simple_LEDP import *

# Create output folder - name changed to reflect new epsilon values and purpose
output_dir = "untxt12" # Changed output directory name
os.makedirs(output_dir, exist_ok=True)

file_pyth = "datasets/wikipedia/squirrel/musae_squirrel_edges.csv"

# Privacy parameters
epsilons = [1, 2] # MODIFIED: Only privacy budgets 1 and 2
delta = 1e-9
eta = 0.4
repeat = 5  # Each experiment repeated once

G_combined = read_squirrel_graph(file_pyth)
# Store experiment results
results = []

# Original graph's densest subgraph (Baseline1), as reference
start_time = time.time()
original_dense_subgraph, original_density = charikar_peeling(G_combined)
baseline1_time = time.time() - start_time

# Test program
for epsilon in epsilons:
    for i in range(repeat): # Changed _ to i for clarity if needed for run-specific seeds later
        print(f"Running experiment for epsilon: {epsilon}, repetition: {i+1}/{repeat}")
        # Baseline 2: Simplified DP algorithm
        start_time = time.time()
        baseline2_dense_subgraph, baseline2_density_ = SEQDENSEDP(G_combined, epsilon, delta)
        baseline2_time = time.time() - start_time
        baseline2_density = validate_subset_density_original(G_combined, baseline2_dense_subgraph)
        baseline2_similarity = compare_subgraph_similarity(original_dense_subgraph, baseline2_dense_subgraph)
        # Calculate ratio for Baseline 2
        ratio_baseline2 = baseline2_density / original_density if original_density != 0 else 0

        # Using new LDP greedy peeling algorithm
        start_time = time.time()
        our_dense_subgraph = ldp_greedy_peeling(G_combined, epsilon)
        our_time = time.time() - start_time
        our_density = validate_subset_density_original(G_combined, our_dense_subgraph)
        our_similarity = compare_subgraph_similarity(original_dense_subgraph, our_dense_subgraph)
        # Calculate ratio for Our Method
        ratio_our = our_density / original_density if original_density != 0 else 0

        # Simple_LEDP
        start_time = time.time()
        simple_dense_subgraph = simple_epsilon_ledp(G_combined, epsilon, eta)
        simple_time = time.time() - start_time
        baseline3_density = validate_subset_density_original(G_combined, simple_dense_subgraph)
        baseline3_similarity = compare_subgraph_similarity(original_dense_subgraph, simple_dense_subgraph)
        # Calculate ratio for Simple_LEDP (Baseline 3)
        ratio_baseline3 = baseline3_density / original_density if original_density != 0 else 0

        # Store results
        results.append({
            "epsilon": epsilon,
            "run_number": i + 1, # Added run number for each experiment
            "original_density": original_density,
            "baseline2_density": baseline2_density,
            "baseline2_similarity": baseline2_similarity,
            "baseline2_time": baseline2_time,
            "ratio_baseline2": ratio_baseline2, # ADDED
            "baseline3_density": baseline3_density,
            "baseline3_similarity": baseline3_similarity,
            "baseline3_time": simple_time,
            "ratio_baseline3": ratio_baseline3, # ADDED
            "our_density": our_density,
            "our_similarity": our_similarity,
            "our_time": our_time,
            "ratio_our": ratio_our, # ADDED
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save detailed results (all runs) to file
detailed_csv_path = os.path.join(output_dir, "squirreldetailed_density_ratios_all_runs.csv")
results_df.to_csv(detailed_csv_path, index=False)
print(f"Detailed results saved to: {detailed_csv_path}")

# --- The following part calculates mean results and plots them ---
# --- This can be kept for immediate feedback or removed if plotting is done separately ---

# Select numeric columns (excluding non-numeric like 'epsilon' if it was object type, though it's fine here)
numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()

# Columns to average (ensure 'epsilon' and 'run_number' are not averaged if not desired)
# For this purpose, we want to average over 'run_number' for each 'epsilon'
cols_to_average = [col for col in numeric_cols if col not in ['epsilon', 'run_number']]


# Group by epsilon and calculate mean, std for the relevant columns
# If repeat > 1, otherwise mean is just the value itself
if repeat > 1:
    mean_results_df = results_df.groupby("epsilon")[cols_to_average].mean().reset_index()
    std_results_df = results_df.groupby("epsilon")[cols_to_average].std().reset_index()
else: # if repeat is 1, mean_results_df is essentially results_df without run_number
    mean_results_df = results_df.drop(columns=['run_number']).reset_index(drop=True)
    # std_results_df would be all NaNs or 0s if repeat is 1, so not very useful.

# Save mean results to file
mean_csv_path = os.path.join(output_dir, "mean_density_ratios.csv")
mean_results_df.to_csv(mean_csv_path, index=False)
print(f"Mean results saved to: {mean_csv_path}")

if repeat > 1:
    std_csv_path = os.path.join(output_dir, "std_density_ratios.csv")
    std_results_df.to_csv(std_csv_path, index=False)
    print(f"Standard deviation results saved to: {std_csv_path}")


# --- Example of plotting ratios from the mean_results_df ---
# This plot will be similar to what the separate plotting script will do with the CSV
plt.figure(figsize=(10, 6))
if "ratio_baseline2" in mean_results_df.columns:
    plt.plot(mean_results_df["epsilon"], mean_results_df["ratio_baseline2"], label="Baseline2 Density Ratio (DP)", marker="s")
if "ratio_our" in mean_results_df.columns:
    plt.plot(mean_results_df["epsilon"], mean_results_df["ratio_our"], label="Our Method Density Ratio (LDP)", marker="^")
if "ratio_baseline3" in mean_results_df.columns:
    plt.plot(mean_results_df["epsilon"], mean_results_df["ratio_baseline3"], label="Simple_LEDP Density Ratio", marker="x")

plt.xlabel("Privacy Budget (epsilon)")
plt.ylabel("Density Ratio (Algorithm Density / Original Density)")
plt.title("Density Ratio vs Privacy Budget (musae_FR)")
plt.xticks([1, 2]) # Ensure ticks for epsilon 1 and 2 are shown
plt.axhline(1, color='grey', linestyle='--', label='Original Density Ratio (Reference)') # Add a line at y=1 for reference
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "ratios_vs_privacy_budget.png"))
plt.show()

print(f"Plots saved in directory: {output_dir}")