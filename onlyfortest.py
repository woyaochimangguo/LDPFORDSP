import os
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from measurements import compare_subgraph_similarity
import pandas as pd
import matplotlib.pyplot as plt
from our_method import *
from ReadGraph import *

# 创建输出文件夹
output_dir = "OUTPUTTEST4000"
os.makedirs(output_dir, exist_ok=True)

# 数据集路径
prefix = './datasets/Facebook/facebook/'
files = ['414.edges', '107.edges', "0.edges", "348.edges", "686.edges", "698.edges", "1684.edges", "1912.edges", "3437.edges", "3980.edges"]

# 隐私参数
epsilons = [0.5, 1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 4, 5, 6, 7, 8, 9, 10]
delta = 1e-9
repeat = 3  # 每个实验重复次数

# 合并所有子图
# G_combined = readFacebook(prefix, files[1])
G_combined = nx.Graph()
for file in files:
    G = readFacebook(prefix, file)
    G_combined = nx.compose(G_combined, G)

# 存储实验结果
results = []

# 测试程序
# 原始图的最密子图（Baseline1），作为参考
original_dense_subgraph, original_density = charikar_peeling(G_combined)

# 测试程序
for epsilon in epsilons:
    for _ in range(repeat):
        # Baseline 2: 简化版 DP 算法
        baseline2_dense_subgraph, baseline2_density = SEQDENSEDP(G_combined, epsilon, delta)
        baseline2_similarity = compare_subgraph_similarity(original_dense_subgraph, baseline2_dense_subgraph)

        # 使用新的 LDP 贪心剥离算法
        our_dense_subgraph = ldp_greedy_peeling(G_combined, epsilon)
        our_density = validate_subset_density_original(G_combined, our_dense_subgraph)
        our_similarity = compare_subgraph_similarity(original_dense_subgraph, our_dense_subgraph)

        # 存储结果
        results.append({
            "epsilon": epsilon,
            "original_density": original_density,  # Baseline1 的密度保持不变
            "baseline2_density": baseline2_density,
            "baseline2_similarity": baseline2_similarity,
            "our_density": our_density,
            "our_similarity": our_similarity,
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

# 保存均值结果到文件
mean_results.to_csv(os.path.join(output_dir, "density_results_with_similarity_mean_test.csv"), index=False)

# 绘图：密度变化 vs 隐私预算
plt.figure(figsize=(10, 6))
plt.plot(mean_results["epsilon"], mean_results["original_density"], label="Original Density (Baseline1)", marker="o")
plt.plot(mean_results["epsilon"], mean_results["baseline2_density"], label="Baseline2 Density (DP)", marker="s")
plt.plot(mean_results["epsilon"], mean_results["our_density"], label="Our Method Density (LDP)", marker="^")
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
plt.xlabel("Privacy Budget (epsilon)")
plt.ylabel("Jaccard Similarity")
plt.title("Jaccard Similarity vs Privacy Budget")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "similarity_vs_privacy_budget_test.png"))
plt.show()