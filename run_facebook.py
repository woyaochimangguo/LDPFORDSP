import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Our_algorithm import *
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from baseline_4_LDP2 import *
from ReadGraph import *
from measurements import *

# 参数设置
prefix = './datasets/Facebook/facebook/'
files = ['414.edges', '107.edges', "0.edges", "348.edges", "686.edges", "698.edges", "1684.edges", "1912.edges", "3437.edges", "3980.edges"]
epsilons = [0.1,  10]
delta = 1e-9
repeat = 1  # 每个实验重复次数

# 创建输出文件夹
output_dir = "output_aggregate"
os.makedirs(output_dir, exist_ok=True)

# 合并所有图
G_combined = nx.Graph()
for file in files:
    G = readFacebook(prefix, file)
    G_combined = nx.compose(G_combined, G)

num_nodes = len(G_combined.nodes)
num_edges = len(G_combined.edges)
graph_info = f"{num_nodes}nodes_{num_edges}edges"

# 将图的邻接表转换为 Baseline 4 使用的格式
adj_list = {node: list(G_combined.neighbors(node)) for node in G_combined.nodes}

results = {
    "epsilons": [],
    "original_density": [],
    "dp_density": [],
    "ldp_density": [],
    "baseline4_density": [],
    "jaccard_baseline1_baseline2": [],
    "jaccard_baseline1_ours": [],
    "jaccard_baseline1_baseline4": [],
    "time_baseline1": [],
    "time_baseline2": [],
    "time_ours": [],
    "time_baseline4": []
}

for epsilon in epsilons:
    original_density_list = []
    dp_density_list = []
    ldp_density_list = []
    baseline4_density_list = []
    jaccard_baseline1_baseline2_list = []
    jaccard_baseline1_ours_list = []
    jaccard_baseline1_baseline4_list = []
    time_baseline1_list = []
    time_baseline2_list = []
    time_ours_list = []
    time_baseline4_list = []

    for _ in range(repeat):
        # Baseline 1
        start_time = time.time()
        dense_subgraph, density = charikar_peeling(G_combined)
        time_baseline1_list.append(time.time() - start_time)
        original_density_list.append(density)

        # Baseline 2
        start_time = time.time()
        dense_subgraph_DP, density_DP = SEQDENSEDP(G_combined, epsilon, delta)
        time_baseline2_list.append(time.time() - start_time)
        dp_density_list.append(density_DP)

        # Ours
        start_time = time.time()
        dense_subgraph_ldpdsp, density_ldpdsp = ldp_charikar_peeling(G_combined, epsilon)
        time_ours_list.append(time.time() - start_time)
        ldp_density_list.append(density_ldpdsp)

        # Baseline 4
        start_time = time.time()
        dense_subgraph_ledp, density_ledp = ledp_densest_subgraph(adj_list, epsilon, eta=0.5)
        time_baseline4_list.append(time.time() - start_time)
        baseline4_density_list.append(density_ledp)

        # Jaccard Similarities
        jaccard_baseline1_baseline2_list.append(compare_subgraph_similarity(dense_subgraph, dense_subgraph_DP))
        jaccard_baseline1_ours_list.append(compare_subgraph_similarity(dense_subgraph, dense_subgraph_ldpdsp))
        jaccard_baseline1_baseline4_list.append(compare_subgraph_similarity(dense_subgraph, dense_subgraph_ledp))

    # 计算平均值
    results["epsilons"].append(epsilon)
    results["original_density"].append(np.mean(original_density_list))
    results["dp_density"].append(np.mean(dp_density_list))
    results["ldp_density"].append(np.mean(ldp_density_list))
    results["baseline4_density"].append(np.mean(baseline4_density_list))
    results["jaccard_baseline1_baseline2"].append(np.mean(jaccard_baseline1_baseline2_list))
    results["jaccard_baseline1_ours"].append(np.mean(jaccard_baseline1_ours_list))
    results["jaccard_baseline1_baseline4"].append(np.mean(jaccard_baseline1_baseline4_list))
    results["time_baseline1"].append(np.mean(time_baseline1_list))
    results["time_baseline2"].append(np.mean(time_baseline2_list))
    results["time_ours"].append(np.mean(time_ours_list))
    results["time_baseline4"].append(np.mean(time_baseline4_list))

# 绘制图像：密度 vs Epsilon
plt.figure(figsize=(10, 6))
plt.plot(results["epsilons"], results["original_density"], label="Baseline1 (Original)", marker='o')
plt.plot(results["epsilons"], results["dp_density"], label="Baseline2 (DP)", marker='s')
plt.plot(results["epsilons"], results["ldp_density"], label="Ours (LDP)", marker='^')
plt.plot(results["epsilons"], results["baseline4_density"], label="Baseline4 (LEDP)", marker='x')
plt.title(f"Density vs Epsilon for Combined Graph: {graph_info}")
plt.xlabel("Epsilon")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
density_plot_path = os.path.join(output_dir, f"density_combined_{graph_info}.png")
plt.savefig(density_plot_path)
plt.close()

# 绘制图像：Jaccard 相似度 vs Epsilon
plt.figure(figsize=(10, 6))
plt.plot(results["epsilons"], results["jaccard_baseline1_baseline2"], label="Jaccard (Baseline2 vs Baseline1)", marker='o')
plt.plot(results["epsilons"], results["jaccard_baseline1_ours"], label="Jaccard (Ours vs Baseline1)", marker='^')
plt.plot(results["epsilons"], results["jaccard_baseline1_baseline4"], label="Jaccard (Baseline4 vs Baseline1)", marker='x')
plt.title(f"Jaccard Similarity vs Epsilon for Combined Graph: {graph_info}")
plt.xlabel("Epsilon")
plt.ylabel("Jaccard Similarity")
plt.legend()
plt.grid(True)
jaccard_plot_path = os.path.join(output_dir, f"jaccard_combined_{graph_info}.png")
plt.savefig(jaccard_plot_path)
plt.close()

# 输出结果
print("\nSummary of results:")
for epsilon, orig, dp, ldp, b4, jac_b1_b2, jac_b1_ours, jac_b1_b4, t_b1, t_b2, t_ours, t_b4 in zip(
        results["epsilons"],
        results["original_density"],
        results["dp_density"],
        results["ldp_density"],
        results["baseline4_density"],
        results["jaccard_baseline1_baseline2"],
        results["jaccard_baseline1_ours"],
        results["jaccard_baseline1_baseline4"],
        results["time_baseline1"],
        results["time_baseline2"],
        results["time_ours"],
        results["time_baseline4"]):
    print(f"Epsilon: {epsilon:.1f}, Orig Density: {orig:.4f}, DP Density: {dp:.4f}, LDP Density: {ldp:.4f}, LEDP Density: {b4:.4f}, "
          f"Jaccard (Baseline2 vs Baseline1): {jac_b1_b2:.4f}, Jaccard (Ours vs Baseline1): {jac_b1_ours:.4f}, Jaccard (Baseline4 vs Baseline1): {jac_b1_b4:.4f}, "
          f"Time (Baseline1): {t_b1:.4f}s, Time (Baseline2): {t_b2:.4f}s, Time (Ours): {t_ours:.4f}s, Time (Baseline4): {t_b4:.4f}s")