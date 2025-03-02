import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 更新文件路径列表，包含新数据集 musae_FR
file_paths = {
    "ca_GrQc": "20241208/ca_GrQc_results/density_results_with_similarity_mean_test.csv",
    "facebook": "20241208/facebook_results/density_results_with_similarity_mean_test.csv",
    "musae_squirrel": "20241208/musae_squirrel_results/density_results_with_similarity_mean_test.csv",
    "musae_ENGB": "20241208/musae_ENGB_results_0_10/density_results_with_similarity_mean_test.csv",
    "musae_DE": "20241208/musae_DE_results/density_results_with_similarity_mean_test.csv",
    "musae_FR": "20241208/musae_FR_results/density_results_with_similarity_mean_test.csv"
}

# 创建输出文件夹
output_draw_dir = "outputdraw_6"
os.makedirs(output_draw_dir, exist_ok=True)

# 初始化图表 (两行三列布局)
fig_density, axes_density = plt.subplots(2, 3, figsize=(30, 16))
fig_similarity, axes_sim = plt.subplots(2, 3, figsize=(30, 16))
fig_time, axes_time = plt.subplots(2, 3, figsize=(30, 16))

# 遍历文件路径
for idx, (dataset, file_path) in enumerate(file_paths.items()):
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        continue

    # 读取结果CSV文件
    mean_results = pd.read_csv(file_path)

    # 提取横坐标范围
    epsilon_range = mean_results["epsilon"].max()

    # 提取运行时间数据
    time_non_private = mean_results.get("baseline1_time", np.zeros_like(mean_results["epsilon"]))
    time_baseline2 = mean_results.get("baseline2_time", np.zeros_like(mean_results["epsilon"]))
    time_our_method = mean_results.get("our_time", np.zeros_like(mean_results["epsilon"]))
    time_baseline3 = mean_results.get("baseline3_time", np.zeros_like(mean_results["epsilon"]))

    # 确定当前子图位置
    row, col = divmod(idx, 3)
    ax_density = axes_density[row, col]
    ax_sim = axes_sim[row, col]
    ax_time = axes_time[row, col]

    # 绘制密度变化图
    ax_density.plot(mean_results["epsilon"], mean_results["original_density"], label="Non-private", marker="o", linestyle='-', antialiased=True)
    ax_density.plot(mean_results["epsilon"], mean_results["baseline2_density"], label="Dhulipala's", marker="^", linestyle='-', antialiased=True)
    ax_density.plot(mean_results["epsilon"], mean_results["our_density"], label="Our's", marker="s", linestyle='-', antialiased=True)
    ax_density.plot(mean_results["epsilon"], mean_results["baseline3_density"], label="Dinitz's", marker="x", linestyle='-', antialiased=True)
    ax_density.set_xlim(0, epsilon_range)
    ax_density.grid(True, linestyle='--', alpha=0.6)
    ax_density.tick_params(axis='both', which='major', labelsize=24, width=2, length=8)
    ax_density.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=28, fontweight='bold')
    ax_density.set_ylabel("Density", fontsize=28, fontweight='bold')
    ax_density.legend(prop={'size': 24, 'weight': 'bold'}, frameon=False, loc="best")
    ax_density.add_patch(plt.Rectangle((0, 1.02), 1, 0.08, transform=ax_density.transAxes, color='lightgray', clip_on=False))
    ax_density.text(0.5, 1.06, dataset, transform=ax_density.transAxes, fontsize=28, fontweight='bold', ha='center', va='center')

    # 绘制集合相似度变化图
    ax_sim.plot(mean_results["epsilon"], mean_results["baseline2_similarity"], label="Dhulipala's", marker="o", linestyle='-', antialiased=True)
    ax_sim.plot(mean_results["epsilon"], mean_results["our_similarity"], label="Our's", marker="^", linestyle='-', antialiased=True)
    ax_sim.plot(mean_results["epsilon"], mean_results["baseline3_similarity"], label="Dinitz's", marker="x", linestyle='-', antialiased=True)
    ax_sim.set_xlim(0, epsilon_range)
    ax_sim.grid(True, linestyle='--', alpha=0.6)
    ax_sim.tick_params(axis='both', which='major', labelsize=24, width=2, length=8)
    ax_sim.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=28, fontweight='bold')
    ax_sim.set_ylabel("Jaccard Similarity", fontsize=28, fontweight='bold')
    ax_sim.legend(prop={'size': 24, 'weight': 'bold'}, frameon=False, loc="best")
    ax_sim.add_patch(plt.Rectangle((0, 1.02), 1, 0.08, transform=ax_sim.transAxes, color='lightgray', clip_on=False))
    ax_sim.text(0.5, 1.06, dataset, transform=ax_sim.transAxes, fontsize=28, fontweight='bold', ha='center', va='center')

    # 绘制运行时间变化图
    ax_time.plot(mean_results["epsilon"], time_non_private, label="Non-private", marker="o", linestyle='-', antialiased=True)
    ax_time.plot(mean_results["epsilon"], time_baseline2, label="Dhulipala's", marker="^", linestyle='-', antialiased=True)
    ax_time.plot(mean_results["epsilon"], time_our_method, label="Our's", marker="s", linestyle='-', antialiased=True)
    ax_time.plot(mean_results["epsilon"], time_baseline3, label="Dinitz's", marker="x", linestyle='-', antialiased=True)
    ax_time.set_xlim(0, epsilon_range)
    ax_time.grid(True, linestyle='--', alpha=0.6)
    ax_time.tick_params(axis='both', which='major', labelsize=24, width=2, length=8)
    ax_time.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=28, fontweight='bold')
    ax_time.set_ylabel("Runtime (ms)", fontsize=28, fontweight='bold')
    ax_time.legend(prop={'size': 24, 'weight': 'bold'}, frameon=False, loc="best")
    ax_time.add_patch(plt.Rectangle((0, 1.02), 1, 0.08, transform=ax_time.transAxes, color='lightgray', clip_on=False))
    ax_time.text(0.5, 1.06, dataset, transform=ax_time.transAxes, fontsize=28, fontweight='bold', ha='center', va='center')

# 调整布局，减少底部空白
fig_density.tight_layout(rect=[0.02, 0.05, 1, 0.95])
fig_similarity.tight_layout(rect=[0.02, 0.05, 1, 0.95])
fig_time.tight_layout(rect=[0.02, 0.05, 1, 0.95])

# 保存图表
fig_density.savefig(os.path.join(output_draw_dir, "density_all_datasets_individual.png"))
fig_similarity.savefig(os.path.join(output_draw_dir, "similarity_all_datasets.png"))
fig_time.savefig(os.path.join(output_draw_dir, "runtime_all_datasets.png"))

plt.show()

print(f"所有图表已成功保存到 {output_draw_dir}")