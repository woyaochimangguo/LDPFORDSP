import os
import time
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from measurements import compare_subgraph_similarity
import pandas as pd
import matplotlib.pyplot as plt
from our_method import *
from ReadGraph import *
from Simple_LEDP import *

# 创建输出文件夹
output_dir = "ca_Astroph_results"
os.makedirs(output_dir, exist_ok=True)

file_pyth = "datasets/CA-AstroPh.txt"

# 隐私参数
epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
delta = 1e-9
eta = 0.4
repeat = 1  # 每个实验重复次数

G_combined = read_ca_astroph(file_pyth)
# 存储实验结果
results = []

# 原始图的最密子图（Baseline1），作为参考
start_time = time.time()
original_dense_subgraph, original_density = charikar_peeling(G_combined)
baseline1_time = time.time() - start_time

# 测试程序
for epsilon in epsilons:
    for _ in range(repeat):
        # Baseline 2: 简化版 DP 算法
        start_time = time.time()
        baseline2_dense_subgraph, baseline2_density_ = SEQDENSEDP(G_combined, epsilon, delta)
        baseline2_time = time.time() - start_time
        baseline2_density = validate_subset_density_original(G_combined, baseline2_dense_subgraph)
        baseline2_similarity = compare_subgraph_similarity(original_dense_subgraph, baseline2_dense_subgraph)

        # 使用新的 LDP 贪心剥离算法
        start_time = time.time()
        our_dense_subgraph = ldp_greedy_peeling(G_combined, epsilon)
        our_time = time.time() - start_time
        our_density = validate_subset_density_original(G_combined, our_dense_subgraph)
        our_similarity = compare_subgraph_similarity(original_dense_subgraph, our_dense_subgraph)

        # Simple_LEDP
        start_time = time.time()
        simple_dense_subgraph = simple_epsilon_ledp(G_combined, epsilon, eta)
        simple_time = time.time() - start_time
        baseline3_density = validate_subset_density_original(G_combined, simple_dense_subgraph)
        baseline3_similarity = compare_subgraph_similarity(original_dense_subgraph, simple_dense_subgraph)

        # 存储结果
        results.append({
            "epsilon": epsilon,
            "original_density": original_density,  # Baseline1 的密度保持不变
            "baseline2_density": baseline2_density,
            "baseline2_similarity": baseline2_similarity,
            "baseline2_time": baseline2_time,
            "baseline3_density": baseline3_density,
            "baseline3_similarity": baseline3_similarity,
            "baseline3_time": simple_time,
            "our_density": our_density,
            "our_similarity": our_similarity,
            "our_time": our_time,
        })

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 筛选数值列（排除非数字列）
numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()

# 确保分组时的列不会重复
if "epsilon" in numeric_cols:
    numeric_cols.remove("epsilon")

# 按 epsilon 分组计算均值，仅针对数值列
mean_results = results_df.groupby("epsilon")[numeric_cols].mean().reset_index()

# 输出结果
print(mean_results)

# 保存详细结果到文件
results_df.to_csv(os.path.join(output_dir, "detailed_density_results_with_subgraphs_test.csv"), index=False)
mean_results.to_csv(os.path.join(output_dir, "density_results_with_similarity_mean_test.csv"), index=False)

# 绘图：密度变化 vs 隐私预算
plt.figure(figsize=(10, 6))
plt.plot(mean_results["epsilon"], mean_results["original_density"], label="Original Density (Baseline1)", marker="o")
plt.plot(mean_results["epsilon"], mean_results["baseline2_density"], label="Baseline2 Density (DP)", marker="s")
plt.plot(mean_results["epsilon"], mean_results["our_density"], label="Our Method Density (LDP)", marker="^")
plt.plot(mean_results["epsilon"], mean_results["baseline3_density"], label="Simple_LEDP", marker="x")
plt.xlabel("Privacy Budget (epsilon)")
plt.ylabel("Density")
plt.title("Density vs Privacy Budget")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "density_vs_privacy_budget_test.png"))
plt.show()

# 绘图：集合相似度变化 vs 隐私预算
plt.figure(figsize=(10, 6))
plt.plot(mean_results["epsilon"], mean_results["baseline2_similarity"], label="Jaccard Similarity (Baseline2 vs Baseline1)", marker="o")
plt.plot(mean_results["epsilon"], mean_results["our_similarity"], label="Jaccard Similarity (Our Method vs Baseline1)", marker="^")
plt.plot(mean_results["epsilon"], mean_results["baseline3_similarity"], label="Jaccard Similarity (Baseline3 vs Baseline1)", marker="x")
plt.xlabel("Privacy Budget (epsilon)")
plt.ylabel("Jaccard Similarity")
plt.title("Jaccard Similarity vs Privacy Budget")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "similarity_vs_privacy_budget_test.png"))
plt.show()

# 绘图：运行时间变化 vs 隐私预算
plt.figure(figsize=(10, 6))
plt.plot(mean_results["epsilon"], [baseline1_time] * len(mean_results), label="Baseline1 Time", linestyle="--", color="red")
plt.plot(mean_results["epsilon"], mean_results["baseline2_time"], label="Baseline2 Time", marker="s", color="orange")
plt.plot(mean_results["epsilon"], mean_results["baseline3_time"], label="Simple_LEDP Time", marker="x", color="blue")
plt.plot(mean_results["epsilon"], mean_results["our_time"], label="Our Method Time", marker="^", color="green")
plt.xlabel("Privacy Budget (epsilon)")
plt.ylabel("Time (seconds)")
plt.title("Execution Time vs Privacy Budget")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "execution_time_vs_privacy_budget_test.png"))
plt.show()