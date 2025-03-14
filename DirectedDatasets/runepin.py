import os
import time
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random
from DirectedGraphBaseline import *
from ReadDirectedGraph import *

# Configure Matplotlib for English labels
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
output_draw_dir = "epininfig_2000_error"
os.makedirs(output_draw_dir, exist_ok=True)

def get_random_subgraph(G, num_nodes=4000):
    """Randomly select a subgraph with the specified number of nodes."""
    selected_nodes = random.sample(list(G.nodes()), num_nodes)
    return G.subgraph(selected_nodes).copy()

# 1. **Load Data**
file_path = 'soc-Epinions1.txt.gz'
G_ori = ReadSocEpinions(file_path)  # Read the full graph
G = get_random_subgraph(G_ori, num_nodes=2000)  # Extract a subgraph

# 2. **Baseline Calculation**
start_time_baseline = time.time()
S_baseline, T_baseline = densest_subgraph_directed(G)
end_time_baseline = time.time()
baseline_runtime = end_time_baseline - start_time_baseline

max_density_baseline = calculate_density(G, S_baseline, T_baseline)
print("\n===== Baseline Results =====")
print(f"Baseline densest subgraph density: {max_density_baseline:.4f}")
print(f"Baseline runtime: {baseline_runtime:.4f} seconds")

# 3. **LDP Calculation for Different ε**
epsilons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []

for epsilon in epsilons:
    start_time_LDP = time.time()
    S_LDP, T_LDP = densest_subgraph_LDP(G, epsilon)  # LDP calculation
    end_time_LDP = time.time()
    LDP_runtime = end_time_LDP - start_time_LDP

    # Calculate LDP density on the original graph
    original_density_LDP = calculate_density(G, S_LDP, T_LDP)

    # Compute Jaccard similarity
    sim_S = jaccard_similarity(S_LDP, S_baseline)
    sim_T = jaccard_similarity(T_LDP, T_baseline)

    results.append((epsilon, original_density_LDP, sim_S, sim_T, LDP_runtime))

    print(f"\n===== ε = {epsilon} =====")
    print(f"LDP densest subgraph density: {original_density_LDP:.4f}")
    print(f"Jaccard similarity of S: {sim_S:.4f}")
    print(f"Jaccard similarity of T: {sim_T:.4f}")
    print(f"LDP runtime: {LDP_runtime:.4f} seconds")

# 4. **Save Results to CSV**
csv_file = os.path.join(output_draw_dir, "results.csv")
df = pd.DataFrame(results, columns=["Epsilon", "LDP Density", "S Similarity", "T Similarity", "LDP Runtime"])
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