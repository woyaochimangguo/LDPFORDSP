import os
import matplotlib.pyplot as plt
import pandas as pd

# 确保输出目录存在
output_dir = "aggregated_figures_real"
os.makedirs(output_dir, exist_ok=True)

# 读取数据集
datasets = {
    "wikifig5": "result/wikifig5/results.csv",
    "cora_10": "result/cora_10/results.csv",
    "bitcoinfig_6": "result/bitcoinfig_6/results.csv"
}

# 统一 Matplotlib 样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义绘图函数，去掉大标题
def plot_subplots(ylabel, filename, y_baseline_col, y_ldp_col, color, marker):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1行3列子图

    for ax, (dataset_name, file_path) in zip(axes, datasets.items()):
        df = pd.read_csv(file_path)

        ax.plot(df["Epsilon"], df[y_baseline_col], 'r--', label="Baseline Density")
        ax.plot(df["Epsilon"], df[y_ldp_col], f'{color}{marker}-', label="LDP")

        ax.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(dataset_name, fontsize=16, fontweight='bold')  # 仅保留子图标题
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
        ax.legend(prop={'size': 12, 'weight': 'bold'}, frameon=False, loc="best")

    plt.tight_layout()  # 移除大标题后，调整布局
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# **绘制 3 张大图**
plot_subplots("Density", "directed-density_comparison.png", "Baseline Density", "LDP Density", 'b', 'o')
plot_subplots("Jaccard Similarity", "directed-similarity_comparison.png", "S Similarity", "T Similarity", 'g', 's')
plot_subplots("Runtime (seconds)", "directed-runtime_comparison.png", "Baseline Runtime", "LDP Runtime", 'm', 'd')

print(f"All aggregated charts saved in {output_dir}")