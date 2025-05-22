import os
import time
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random
from DirectedGraphBaseline import *
from ReadDirectedGraph import *

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

output_draw_dir = "airports_1248"
os.makedirs(output_draw_dir, exist_ok=True)

datasets_dir = '/Users/teco/Documents/GitHub/LDPFORDSP/DirectedDatasets/datasets/'
file_path = '/Users/teco/Documents/GitHub/LDPFORDSP/DirectedDatasets/datasets/airports.txt'
G = read_graph(file_path)   # Read the full graph
# 2. Baseline
start_time_baseline = time.time()
S_baseline, T_baseline = densest_subgraph_directed(G)
end_time_baseline = time.time()
baseline_runtime = end_time_baseline - start_time_baseline
max_density_baseline = calculate_density(G, S_baseline, T_baseline)

print("\n===== Baseline Results =====")
print(f"Baseline densest subgraph density: {max_density_baseline:.4f}")
print(f"Baseline runtime: {baseline_runtime:.4f} seconds")

# 3. LDP runs (with 5 repetitions per epsilon)
epsilons = [1, 2, 4, 8]
repeat_times = 5
summary_results = []
all_runs = []

for epsilon in epsilons:
    densities, sim_S_list, sim_T_list, runtimes = [], [], [], []

    for run in range(repeat_times):
        start_time_LDP = time.time()
        S_LDP, T_LDP = densest_subgraph_LDP(G, epsilon)
        end_time_LDP = time.time()
        LDP_runtime = end_time_LDP - start_time_LDP

        density = calculate_density(G, S_LDP, T_LDP)
        sim_S = jaccard_similarity(S_LDP, S_baseline)
        sim_T = jaccard_similarity(T_LDP, T_baseline)

        # 单次结果添加到完整记录
        all_runs.append({
            "Epsilon": epsilon,
            "Run": run + 1,
            "Density": density,
            "S Similarity": sim_S,
            "T Similarity": sim_T,
            "Runtime": LDP_runtime
        })

        # 添加到临时列表用于平均
        densities.append(density)
        sim_S_list.append(sim_S)
        sim_T_list.append(sim_T)
        runtimes.append(LDP_runtime)

    # 汇总每个 epsilon 的平均值
    summary_results.append((
        epsilon,
        sum(densities) / repeat_times,
        sum(sim_S_list) / repeat_times,
        sum(sim_T_list) / repeat_times,
        sum(runtimes) / repeat_times
    ))

    print(f"\n===== ε = {epsilon} =====")
    print(f"Average LDP density: {summary_results[-1][1]:.4f}")
    print(f"Average Jaccard similarity S: {summary_results[-1][2]:.4f}")
    print(f"Average Jaccard similarity T: {summary_results[-1][3]:.4f}")
    print(f"Average runtime: {summary_results[-1][4]:.4f} seconds")

# 4. Save all runs
df_all = pd.DataFrame(all_runs)
df_all.to_csv(os.path.join(output_draw_dir, "all_runs.csv"), index=False)

# 5. Save summary
df_summary = pd.DataFrame(summary_results, columns=["Epsilon", "LDP Density", "S Similarity", "T Similarity", "LDP Runtime"])
df_summary["Baseline Density"] = max_density_baseline
df_summary["Baseline Runtime"] = baseline_runtime
df_summary.to_csv(os.path.join(output_draw_dir, "summary_results.csv"), index=False)

# 6. Plot
epsilons = [r[0] for r in summary_results]

# Density plot
plt.figure(figsize=(10, 6))
plt.plot(epsilons, [max_density_baseline] * len(epsilons), 'r--', label="Baseline Density")
plt.plot(epsilons, [r[1] for r in summary_results], 'bo-', label="LDP Subgraph Density")
plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
plt.ylabel("Density", fontsize=18, fontweight='bold')
plt.title("Privacy Budget vs. Subgraph Density", fontsize=20, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
plt.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")
plt.savefig(os.path.join(output_draw_dir, "density_comparison.png"))
plt.show()

# Similarity plot
plt.figure(figsize=(10, 6))
plt.plot(epsilons, [r[2] for r in summary_results], 'go-', label="S Jaccard Similarity")
plt.plot(epsilons, [r[3] for r in summary_results], 'mo-', label="T Jaccard Similarity")
plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
plt.ylabel("Jaccard Similarity", fontsize=18, fontweight='bold')
plt.title("Privacy Budget vs. Set Similarity", fontsize=20, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
plt.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")
plt.savefig(os.path.join(output_draw_dir, "similarity_comparison.png"))
plt.show()

# Runtime plot
plt.figure(figsize=(10, 6))
plt.plot(epsilons, [baseline_runtime] * len(epsilons), 'r--', label="Baseline Runtime")
plt.plot(epsilons, [r[4] for r in summary_results], 'bo-', label="LDP Runtime")
plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
plt.ylabel("Runtime (seconds)", fontsize=18, fontweight='bold')
plt.title("Privacy Budget vs. Runtime", fontsize=20, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
plt.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")
plt.savefig(os.path.join(output_draw_dir, "runtime_comparison.png"))
plt.show()

print(f"All charts and CSV files saved to {output_draw_dir}")
