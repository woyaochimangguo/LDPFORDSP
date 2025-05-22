import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from DirectedGraphBaseline import *
from ReadDirectedGraph import *

# Configure Matplotlib for English labels
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
output_draw_dir = "rattus_fig"
os.makedirs(output_draw_dir, exist_ok=True)

# 1. **Load Data**
file_path = '/Users/teco/Documents/GitHub/LDPFORDSP/DirectedDatasets/datasets/rattus.txt'
G = read_graph(file_path)  # Read the full graph

# 2. **Baseline Calculation**
start_time_baseline = time.time()
S_baseline, T_baseline = densest_subgraph_directed(G)
end_time_baseline = time.time()
baseline_runtime = end_time_baseline - start_time_baseline

max_density_baseline = calculate_density(G, S_baseline, T_baseline)
print("\n===== Baseline Results =====")
print(f"Baseline densest subgraph density: {max_density_baseline:.4f}")
print(f"Baseline runtime: {baseline_runtime:.4f} seconds")

# 3. **LDP Calculation for Different ε with repetition**
epsilons = [0.1,1,2,3,4,5,6,7,8,9,10]
repeat_times = 1
results = []

for epsilon in epsilons:
    densities, sims_S, sims_T, runtimes = [], [], [], []

    for _ in range(repeat_times):
        start_time_LDP = time.time()
        S_LDP, T_LDP = densest_subgraph_LDP(G, epsilon)  # LDP calculation
        end_time_LDP = time.time()

        # Calculate LDP density on the original graph
        densities.append(calculate_density(G, S_LDP, T_LDP))

        # Compute Jaccard similarity
        sims_S.append(jaccard_similarity(S_LDP, S_baseline))
        sims_T.append(jaccard_similarity(T_LDP, T_baseline))

        runtimes.append(end_time_LDP - start_time_LDP)

    avg_density = sum(densities) / repeat_times
    avg_sim_S = sum(sims_S) / repeat_times
    avg_sim_T = sum(sims_T) / repeat_times
    avg_runtime = sum(runtimes) / repeat_times

    results.append((epsilon, avg_density, avg_sim_S, avg_sim_T, avg_runtime))

    print(f"\n===== ε = {epsilon} =====")
    print(f"Avg LDP densest subgraph density: {avg_density:.4f}")
    print(f"Avg Jaccard similarity of S: {avg_sim_S:.4f}")
    print(f"Avg Jaccard similarity of T: {avg_sim_T:.4f}")
    print(f"Avg LDP runtime: {avg_runtime:.4f} seconds")

# 4. **Save Results to CSV**
csv_file = os.path.join(output_draw_dir, "results.csv")
df = pd.DataFrame(results, columns=["Epsilon", "Avg LDP Density", "Avg S Similarity", "Avg T Similarity", "Avg LDP Runtime"])
df["Baseline Density"] = max_density_baseline
df["Baseline Runtime"] = baseline_runtime
df.to_csv(csv_file, index=False)
print(f"\nResults saved to {csv_file}")

# 5. **Plot Density Comparison**
plt.figure(figsize=(10, 6))
plt.plot(epsilons, [max_density_baseline] * len(epsilons), 'r--', label="Baseline Density")  # Removed redundant linestyle and marker
plt.plot(epsilons, [r[1] for r in results], 'bo-', label="LDP Subgraph Density")  # Removed redundant linestyle and marker
plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
plt.ylabel("Density", fontsize=18, fontweight='bold')
plt.title("Privacy Budget vs. Subgraph Density", fontsize=20, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
plt.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")
plt.savefig(os.path.join(output_draw_dir, "directed-density_comparison.png"))
plt.show()

# 6. **Plot Jaccard Similarity Comparison**
plt.figure(figsize=(10, 6))
plt.plot(epsilons, [r[2] for r in results], 'go-', label="S Jaccard Similarity")  # Removed redundant linestyle and marker
plt.plot(epsilons, [r[3] for r in results], 'mo-', label="T Jaccard Similarity")  # Removed redundant linestyle and marker
plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
plt.ylabel("Jaccard Similarity", fontsize=18, fontweight='bold')
plt.title("Privacy Budget vs. Set Similarity", fontsize=20, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
plt.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")
plt.savefig(os.path.join(output_draw_dir, "directed-similarity_comparison.png"))
plt.show()

# 7. **Plot Runtime Comparison**
plt.figure(figsize=(10, 6))
plt.plot(epsilons, [baseline_runtime] * len(epsilons), 'r--', label="Baseline Runtime")  # Removed redundant linestyle and marker
plt.plot(epsilons, [r[4] for r in results], 'bo-', label="LDP Runtime")  # Removed redundant linestyle and marker
plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
plt.ylabel("Runtime (seconds)", fontsize=18, fontweight='bold')
plt.title("Privacy Budget vs. Runtime", fontsize=20, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
plt.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")
plt.savefig(os.path.join(output_draw_dir, "directed-runtime_comparison.png"))
plt.show()

print(f"All charts successfully saved to {output_draw_dir}")
