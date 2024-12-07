import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from our_method import *
from ReadGraph import *
import os
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from measurements import compare_subgraph_similarity
import pandas as pd
import matplotlib.pyplot as plt
from our_method import *
from ReadGraph import *
from Simple_LEDP import *
from kcoredespbaseline import *
from baseline_3_LDP import *

# 测试代码

# 创建输出文件夹
output_dir = "OUTPUTTEST_ALGORITHM_80"
os.makedirs(output_dir, exist_ok=True)

# 数据集路径
prefix = './datasets/Facebook/facebook/'
files = ['414.edges', '107.edges', "0.edges", "348.edges", "686.edges", "698.edges", "1684.edges", "1912.edges", "3437.edges", "3980.edges"]

# 隐私参数
epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3,5, 4, 4.5]
eta = 0.4
repeat = 1  # 每个实验重复次数

# 构建图
# G_combined = readFacebook(prefix, files[0])
G_combined = nx.Graph()
for file in files:
    G = readFacebook(prefix, file)
    G_combined = nx.compose(G_combined, G)
# 原始图的最密子图（Baseline1），作为参考
original_dense_subgraph, original_density = charikar_peeling(G_combined)

# 存储实验结果
results = []

for epsilon in epsilons:
    for _ in range(repeat):
        # Baseline2: Simple ε-LEDP Algorithm
        baseline_simple_dense_subgraph = simple_epsilon_ledp(G_combined.copy(), total_epsilon=epsilon, eta=eta)
        baseline_simple_density = validate_subset_density_original(G_combined, baseline_simple_dense_subgraph)
        baseline_simple_similarity = compare_subgraph_similarity(original_dense_subgraph, baseline_simple_dense_subgraph)

        # Baseline3: Algorithm 8
        baseline_8_dense_subgraph = algorithm_8_densest_subgraph(G_combined.copy(), epsilon)
        baseline_8_density = validate_subset_density_original(G_combined, baseline_8_dense_subgraph)
        baseline_8_similarity = compare_subgraph_similarity(original_dense_subgraph, baseline_8_dense_subgraph)

        # Baseline4: Algorithm
        baseline_4_dense_subgraph = ledp_densest_subgraph(G_combined.copy(), epsilon,0.3)
        baseline_4_density = validate_subset_density_original(G_combined, baseline_4_dense_subgraph)
        baseline_4_similarity = compare_subgraph_similarity(original_dense_subgraph, baseline_4_dense_subgraph)

        # Our Method
        our_dense_subgraph = ldp_greedy_peeling(G_combined.copy(), epsilon)
        our_density = validate_subset_density_original(G_combined, our_dense_subgraph)
        our_similarity = compare_subgraph_similarity(original_dense_subgraph, our_dense_subgraph)

        # 存储结果
        results.append({
            "epsilon": epsilon,
            "original_density": original_density,  # Baseline1 的密度保持不变
            "baseline_simple_density": baseline_simple_density,
            "baseline_simple_similarity": baseline_simple_similarity,
            "baseline_8_density": baseline_8_density,
            "baseline_8_similarity": baseline_8_similarity,
            "baseline_4_density": baseline_4_density,
            "baseline_4_similarity": baseline_4_similarity,
            "our_density": our_density,
            "our_similarity": our_similarity,
        })

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 按 epsilon 分组计算均值
mean_results = results_df.groupby("epsilon").mean().reset_index()

# 输出结果
print(mean_results)

# 保存详细结果到文件
results_df.to_csv(os.path.join(output_dir, "detailed_density_results_test.csv"), index=False)
mean_results.to_csv(os.path.join(output_dir, "density_results_mean_test.csv"), index=False)

# 绘图：密度变化 vs 隐私预算
plt.figure(figsize=(10, 6))
plt.plot(mean_results["epsilon"], mean_results["original_density"], label="Original Density (Baseline1)", marker="o")
plt.plot(mean_results["epsilon"], mean_results["baseline_simple_density"], label="Baseline2 Density (Simple LEDP)", marker="s")
plt.plot(mean_results["epsilon"], mean_results["baseline_8_density"], label="Baseline3 Density (Algorithm 8)", marker="d")
plt.plot(mean_results["epsilon"], mean_results["baseline_4_density"], label="Baseline4 Density (Algorithm)", marker="x")
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
plt.plot(mean_results["epsilon"], mean_results["baseline_simple_similarity"], label="Jaccard Similarity (Baseline2 vs Baseline1)", marker="o")
plt.plot(mean_results["epsilon"], mean_results["baseline_8_similarity"], label="Jaccard Similarity (Baseline3 vs Baseline1)", marker="d")
plt.plot(mean_results["epsilon"], mean_results["baseline_4_similarity"], label="Jaccard Similarity (Baseline4 vs Baseline1)", marker="x")
plt.plot(mean_results["epsilon"], mean_results["our_similarity"], label="Jaccard Similarity (Our Method vs Baseline1)", marker="^")
plt.xlabel("Privacy Budget (epsilon)")
plt.ylabel("Jaccard Similarity")
plt.title("Jaccard Similarity vs Privacy Budget")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "similarity_vs_privacy_budget_test.png"))
plt.show()