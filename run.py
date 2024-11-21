import os
from Our_algorithm import *
from datetime import datetime
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from ReadGraph import *
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from measurements import *
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
prefix = './datasets/Facebook/facebook/'
files = ['414.edges', '107.edges', "0.edges", "348.edges", "686.edges", "698.edges", "1684.edges", "1912.edges", "3437.edges", "3980.edges"]
epsilons = [0.1,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
delta = 1e-9
repeat = 10  # 每个实验重复次数

# 创建输出文件夹
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 结果存储
all_results = {}

# 实验
for file in files:
    print(f"\nProcessing graph: {file}")
    G = readFacebook(prefix, file)

    results_for_file = {
        "epsilons": [],
        "original_density": [],
        "dp_density": [],
        "ldp_density": [],
        "jaccard_baseline1_baseline2": [],
        "jaccard_baseline1_ours": []
    }

    for epsilon in epsilons:
        original_density_list = []
        dp_density_list = []
        ldp_density_list = []
        jaccard_baseline1_baseline2_list = []
        jaccard_baseline1_ours_list = []

        for _ in range(repeat):
            # 原始图的稠密子图提取
            dense_subgraph, density = charikar_peeling(G)
            original_density_list.append(density)

            # 中心化差分隐私的稠密子图提取
            dense_subgraph_DP, density_DP = SEQDENSEDP(G, epsilon, delta)
            dp_density_list.append(density_DP)

            # 计算 Jaccard 相似度（baseline2 vs baseline1）
            jaccard_baseline1_baseline2 = compare_subgraph_similarity(dense_subgraph, dense_subgraph_DP)
            jaccard_baseline1_baseline2_list.append(jaccard_baseline1_baseline2)

            # 我们的算法差分隐私后的噪声图的稠密子图提取
            dense_subgraph_ldpdsp, density_ldpdsp = ldp_charikar_peeling(G, epsilon)
            ldp_density_list.append(density_ldpdsp)

            # 计算 Jaccard 相似度（ours vs baseline1）
            jaccard_baseline1_ours = compare_subgraph_similarity(dense_subgraph, dense_subgraph_ldpdsp)
            jaccard_baseline1_ours_list.append(jaccard_baseline1_ours)

        # 计算平均值
        avg_original_density = np.mean(original_density_list)
        avg_dp_density = np.mean(dp_density_list)
        avg_ldp_density = np.mean(ldp_density_list)
        avg_jaccard_baseline1_baseline2 = np.mean(jaccard_baseline1_baseline2_list)
        avg_jaccard_baseline1_ours = np.mean(jaccard_baseline1_ours_list)

        # 存储结果
        results_for_file["epsilons"].append(epsilon)
        results_for_file["original_density"].append(avg_original_density)
        results_for_file["dp_density"].append(avg_dp_density)
        results_for_file["ldp_density"].append(avg_ldp_density)
        results_for_file["jaccard_baseline1_baseline2"].append(avg_jaccard_baseline1_baseline2)
        results_for_file["jaccard_baseline1_ours"].append(avg_jaccard_baseline1_ours)

    all_results[file] = results_for_file
    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(results_for_file["epsilons"], results_for_file["original_density"], label="Baseline1 (Original)", marker='o')
    plt.plot(results_for_file["epsilons"], results_for_file["dp_density"], label="Baseline2 (DP)", marker='s')
    plt.plot(results_for_file["epsilons"], results_for_file["ldp_density"], label="Ours (LDP)", marker='^')
    plt.title(f"Density vs Epsilon for Graph: {file}")
    plt.xlabel("Epsilon")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    density_plot_path = os.path.join(output_dir, f"{file}_density_vs_epsilon.png")
    plt.savefig(density_plot_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(results_for_file["epsilons"], results_for_file["jaccard_baseline1_baseline2"], label="Jaccard (Baseline2 vs Baseline1)", marker='o')
    plt.plot(results_for_file["epsilons"], results_for_file["jaccard_baseline1_ours"], label="Jaccard (Ours vs Baseline1)", marker='^')
    plt.title(f"Jaccard Similarity vs Epsilon for Graph: {file}")
    plt.xlabel("Epsilon")
    plt.ylabel("Jaccard Similarity")
    plt.legend()
    plt.grid(True)
    jaccard_plot_path = os.path.join(output_dir, f"{file}_jaccard_vs_epsilon.png")
    plt.savefig(jaccard_plot_path)
    plt.close()

# 总结输出
print("\nSummary of results:")
for file, result in all_results.items():
    print(f"Graph: {file}")
    for epsilon, orig, dp, ldp, jac_b1_b2, jac_b1_ours in zip(
            result["epsilons"],
            result["original_density"],
            result["dp_density"],
            result["ldp_density"],
            result["jaccard_baseline1_baseline2"],
            result["jaccard_baseline1_ours"]):
        print(f"Epsilon: {epsilon:.1f}, Orig Density: {orig:.4f}, DP Density: {dp:.4f}, LDP Density: {ldp:.4f}, "
              f"Jaccard (Baseline2 vs Baseline1): {jac_b1_b2:.4f}, Jaccard (Ours vs Baseline1): {jac_b1_ours:.4f}")
    print("-" * 50)