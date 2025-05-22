import os
import matplotlib.pyplot as plt
import pandas as pd

# 确保输出目录存在
output_dir = "aggregated_figures_real"
os.makedirs(output_dir, exist_ok=True)

# 各数据集 CSV 路径，其中 bitcoin 和 epin 的 CSV 无 header，其它数据集含有 header
datasets = {
    "airports": "directed_result/airports/results.csv",
    "aucs": "directed_result/aucs_fig/results.csv",
    "bitcoin": "directed_result/bitcoinfig/results.csv",
    "epin": "directed_result/epininfig/results.csv",
    "htmlconf": "directed_result/htmlconf_fig/results.csv",
    "rattus": "directed_result/rattus_fig/results.csv",
}

# 统一列名（与生成 CSV 时顺序一致）
col_names = [
    "Epsilon",
    "LDP Density",
    "S Similarity",
    "T Similarity",
    "LDP Runtime",
    "Baseline Density",
    "Baseline Runtime"
]

def read_dataset(dataset_name, file_path):
    """
    对于 bitcoin 和 epin，由于 CSV 无 header，直接指定列名；
    对于其他数据集，若列名为 “Avg LDP Density”等，则重命名为统一名称。
    """
    if dataset_name in []:
        df = pd.read_csv(file_path, header=None, names=col_names)
    else:
        df = pd.read_csv(file_path)
        # 如果数据集中使用了 "Avg ..." 前缀，则重命名为统一的列名
        if "LDP Density" not in df.columns and "Avg LDP Density" in df.columns:
            df.rename(columns={"Avg LDP Density": "LDP Density"}, inplace=True)
        if "S Similarity" not in df.columns and "Avg S Similarity" in df.columns:
            df.rename(columns={"Avg S Similarity": "S Similarity"}, inplace=True)
        if "T Similarity" not in df.columns and "Avg T Similarity" in df.columns:
            df.rename(columns={"Avg T Similarity": "T Similarity"}, inplace=True)
        if "LDP Runtime" not in df.columns and "Avg LDP Runtime" in df.columns:
            df.rename(columns={"Avg LDP Runtime": "LDP Runtime"}, inplace=True)
    # 转换 Epsilon 为数值型
    df["Epsilon"] = pd.to_numeric(df["Epsilon"], errors="coerce")
    return df

def plot_density_subplots(datasets, output_dir, filename):
    """
    绘制密度比较图：
      - x 轴为 Epsilon；
      - 红色虚线表示 Baseline Density（常数，取第一行的值）；
      - 蓝色实线表示 LDP Density。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (dataset_name, file_path) in zip(axes, datasets.items()):
        df = read_dataset(dataset_name, file_path)
        # baseline 密度取第一行（假设各行一致）
        baseline_density = df["Baseline Density"].iloc[0]
        ax.plot(df["Epsilon"], [baseline_density] * len(df), 'r--', label="Baseline Density")
        ax.plot(df["Epsilon"], df["LDP Density"], 'bo-', label="LDP Subgraph Density")

        ax.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
        ax.set_ylabel("Density", fontsize=18, fontweight='bold')
        ax.set_title(dataset_name, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
        ax.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_similarity_subplots(datasets, output_dir, filename):
    """
    绘制相似度比较图：
      - x 轴为 Epsilon；
      - 绿色实线表示 S Similarity；
      - 品红实线表示 T Similarity。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (dataset_name, file_path) in zip(axes, datasets.items()):
        df = read_dataset(dataset_name, file_path)
        ax.plot(df["Epsilon"], df["S Similarity"], 'go-', label="S Jaccard Similarity")
        ax.plot(df["Epsilon"], df["T Similarity"], 'mo-', label="T Jaccard Similarity")

        ax.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
        ax.set_ylabel("Jaccard Similarity", fontsize=18, fontweight='bold')
        ax.set_title(dataset_name, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
        ax.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_runtime_subplots(datasets, output_dir, filename):
    """
    绘制运行时间比较图：
      - x 轴为 Epsilon；
      - 红色虚线表示 Baseline Runtime（常数，取第一行的值）；
      - 蓝色实线表示 LDP Runtime。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (dataset_name, file_path) in zip(axes, datasets.items()):
        df = read_dataset(dataset_name, file_path)
        baseline_runtime = df["Baseline Runtime"].iloc[0]
        ax.plot(df["Epsilon"], [baseline_runtime] * len(df), 'r--', label="Baseline Runtime")
        ax.plot(df["Epsilon"], df["LDP Runtime"], 'bo-', label="LDP Runtime")

        ax.set_xlabel(r"Epsilon ($\epsilon$)", fontsize=18, fontweight='bold')
        ax.set_ylabel("Runtime (seconds)", fontsize=18, fontweight='bold')
        ax.set_title(dataset_name, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
        ax.legend(prop={'size': 14, 'weight': 'bold'}, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# 分别绘制密度、相似度和运行时间的聚合图
plot_density_subplots(datasets, output_dir, "directed-density_comparison.png")
plot_similarity_subplots(datasets, output_dir, "directed-similarity_comparison.png")
plot_runtime_subplots(datasets, output_dir, "directed-runtime_comparison.png")

print(f"All aggregated charts saved in {output_dir}")