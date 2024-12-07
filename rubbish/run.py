import os
import time
import numpy as np
import random
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
epsilons = [ 0.5,1,1.5,2, 3,  5,  8, 10]
delta = 1e-9
repeat = 10 # 每个实验重复次数

# 创建输出文件夹
output_dir = "output_1"
os.makedirs(output_dir, exist_ok=True)

all_results = {
    "node_counts": [],        # 每个图的节点数量
    "time_baseline1": [],     # Baseline1 平均运行时间
    "time_baseline2": [],     # Baseline2 平均运行时间
    "time_ours": [],          # Ours 平均运行时间
    "time_baseline4": []      # Baseline4 平均运行时间
}

# 结果存储
#all_results = {}
# 实验
for file in files:
    print(f"\nProcessing graph: {file}")
    G = readFacebook(prefix, file)
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    graph_info = f"{num_nodes}nodes_{num_edges}edges"

    # 将图的邻接表转换为 Baseline 4 使用的格式
    adj_list = {node: list(G.neighbors(node)) for node in G.nodes}

    results_for_file = {
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
        "time_baseline4": [],
    }

    for epsilon in epsilons:
        # 初始化结果存储列表
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
            dense_subgraph, density = charikar_peeling(G)
            time_baseline1_list.append(time.time() - start_time)
            original_density_list.append(density)

            # Baseline 2
            start_time = time.time()
            dense_subgraph_DP, density_DP = SEQDENSEDP(G, epsilon, delta)
            time_baseline2_list.append(time.time() - start_time)
            dp_density_list.append(density_DP)

            # Ours
            start_time = time.time()
            dense_subgraph_ldpdsp, density_ldpdsp = ldp_charikar_peeling_distributed(G, epsilon)
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
        avg_original_density = np.mean(original_density_list)
        avg_dp_density = np.mean(dp_density_list)
        avg_ldp_density = np.mean(ldp_density_list)
        avg_baseline4_density = np.mean(baseline4_density_list)
        avg_jaccard_baseline1_baseline2 = np.mean(jaccard_baseline1_baseline2_list)
        avg_jaccard_baseline1_ours = np.mean(jaccard_baseline1_ours_list)
        avg_jaccard_baseline1_baseline4 = np.mean(jaccard_baseline1_baseline4_list)
        avg_time_baseline1 = np.mean(time_baseline1_list)
        avg_time_baseline2 = np.mean(time_baseline2_list)
        avg_time_ours = np.mean(time_ours_list)
        avg_time_baseline4 = np.mean(time_baseline4_list)
        # 存储结果
        results_for_file["epsilons"].append(epsilon)
        results_for_file["original_density"].append(avg_original_density)
        results_for_file["dp_density"].append(avg_dp_density)
        results_for_file["ldp_density"].append(avg_ldp_density)
        results_for_file["baseline4_density"].append(avg_baseline4_density)
        results_for_file["jaccard_baseline1_baseline2"].append(avg_jaccard_baseline1_baseline2)
        results_for_file["jaccard_baseline1_ours"].append(avg_jaccard_baseline1_ours)
        results_for_file["jaccard_baseline1_baseline4"].append(avg_jaccard_baseline1_baseline4)
        results_for_file["time_baseline1"].append(avg_time_baseline1)
        results_for_file["time_baseline2"].append(avg_time_baseline2)
        results_for_file["time_ours"].append(avg_time_ours)
        results_for_file["time_baseline4"].append(avg_time_baseline4)

        # 全局结果更新
    all_results["node_counts"].append(num_nodes)
    all_results["time_baseline1"].append(np.mean(results_for_file["time_baseline1"]))
    all_results["time_baseline2"].append(np.mean(results_for_file["time_baseline2"]))
    all_results["time_ours"].append(np.mean(results_for_file["time_ours"]))
    all_results["time_baseline4"].append(np.mean(results_for_file["time_baseline4"]))
    all_results[file] = results_for_file

    # 绘制图像：密度 vs Epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(results_for_file["epsilons"], results_for_file["original_density"], label="Baseline1 (Original)", marker='o')
    plt.plot(results_for_file["epsilons"], results_for_file["dp_density"], label="Baseline2 (DP)", marker='s')
    plt.plot(results_for_file["epsilons"], results_for_file["ldp_density"], label="Ours (LDP)", marker='^')
    plt.plot(results_for_file["epsilons"], results_for_file["baseline4_density"], label="Baseline4 (LEDP)", marker='x')
    plt.title(f"Density vs Epsilon for Graph: {graph_info}")
    plt.xlabel("Epsilon")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    density_plot_path = os.path.join(output_dir, f"density_{graph_info}.png")
    plt.savefig(density_plot_path)
    plt.close()

    # 绘制图像：Jaccard 相似度 vs Epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(results_for_file["epsilons"], results_for_file["jaccard_baseline1_baseline2"], label="Jaccard (Baseline2 vs Baseline1)", marker='o')
    plt.plot(results_for_file["epsilons"], results_for_file["jaccard_baseline1_ours"], label="Jaccard (Ours vs Baseline1)", marker='^')
    plt.plot(results_for_file["epsilons"], results_for_file["jaccard_baseline1_baseline4"], label="Jaccard (Baseline4 vs Baseline1)", marker='x')
    plt.title(f"Jaccard Similarity vs Epsilon for Graph: {graph_info}")
    plt.xlabel("Epsilon")
    plt.ylabel("Jaccard Similarity")
    plt.legend()
    plt.grid(True)
    jaccard_plot_path = os.path.join(output_dir, f"jaccard_{graph_info}.png")
    plt.savefig(jaccard_plot_path)
    plt.close()

    # 绘制图像：运行时间 vs 节点数量
    plt.figure(figsize=(10, 6))
    plt.plot(all_results["node_counts"], all_results["time_baseline1"], label="Baseline1 Time", marker='o')
    plt.plot(all_results["node_counts"], all_results["time_baseline2"], label="Baseline2 Time", marker='s')
    plt.plot(all_results["node_counts"], all_results["time_ours"], label="Ours Time", marker='^')
    plt.plot(all_results["node_counts"], all_results["time_baseline4"], label="Baseline4 Time", marker='x')
    plt.title("Runtime vs Node Count")
    plt.xlabel("Node Count")
    plt.ylabel("Average Runtime (s)")
    plt.legend()
    plt.grid(True)
    runtime_plot_path = os.path.join(output_dir, "runtime_vs_node_count.png")
    plt.savefig(runtime_plot_path)
    plt.close()

# 输出结果
print("\nSummary of results:")
for file, result in all_results.items():
    print(f"Graph: {file}")
    for epsilon, orig, dp, ldp, b4, jac_b1_b2, jac_b1_ours, jac_b1_b4, t_b1, t_b2, t_ours, t_b4 in zip(
            result["epsilons"],
            result["original_density"],
            result["dp_density"],
            result["ldp_density"],
            result["baseline4_density"],
            result["jaccard_baseline1_baseline2"],
            result["jaccard_baseline1_ours"],
            result["jaccard_baseline1_baseline4"],
            result["time_baseline1"],
            result["time_baseline2"],
            result["time_ours"],
            result["time_baseline4"]):
        print(f"Epsilon: {epsilon:.1f}, Orig Density: {orig:.4f}, DP Density: {dp:.4f}, LDP Density: {ldp:.4f}, LEDP Density: {b4:.4f}, "
              f"Jaccard (Baseline2 vs Baseline1): {jac_b1_b2:.4f}, Jaccard (Ours vs Baseline1): {jac_b1_ours:.4f}, Jaccard (Baseline4 vs Baseline1): {jac_b1_b4:.4f}, "
              f"Time (Baseline1): {t_b1:.4f}s, Time (Baseline2): {t_b2:.4f}s, Time (Ours): {t_ours:.4f}s, Time (Baseline4): {t_b4:.4f}s")
    print("-" * 50)


