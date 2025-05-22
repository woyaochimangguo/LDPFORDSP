import os
import time
import pandas as pd
from DirectedGraphBaseline import *
from ReadDirectedGraph import *

# 数据集路径和输出路径设置
datasets_dir = '/Users/teco/Documents/GitHub/LDPFORDSP/DirectedDatasets/datasets/'
output_base_dir = "txt12"
os.makedirs(output_base_dir, exist_ok=True)

# 隐私预算设置
epsilons = [1, 2]
repeat_times = 5

# 遍历datasets目录下的所有txt文件
for filename in sorted(os.listdir(datasets_dir)):
    if filename.endswith('.txt'):
        file_path = os.path.join(datasets_dir, filename)
        dataset_name = os.path.splitext(filename)[0]
        output_draw_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(output_draw_dir, exist_ok=True)

        print(f"\nProcessing dataset: {filename}")

        G = read_graph(file_path)  # 读取图

        # 计算baseline
        S_baseline, T_baseline = densest_subgraph_directed(G)
        baseline_density = calculate_density(G, S_baseline, T_baseline)
        initial_density = calculate_density(G, list(G.nodes), list(G.nodes))  # 计算初始密度

        # 收集当前数据集的实验结果
        dataset_results = {
            "Epsilon": [],
            "Run": [],
            "Initial Density": [],
            "LDP Density": [],
            "Baseline Density": []
        }

        # 针对每个隐私预算进行实验
        for epsilon in epsilons:
            for run in range(repeat_times):
                start_time = time.time()
                S_LDP, T_LDP = densest_subgraph_LDP(G, epsilon)
                end_time = time.time()

                density_ldp = calculate_density(G, S_LDP, T_LDP)

                # 记录结果
                dataset_results["Epsilon"].append(epsilon)
                dataset_results["Run"].append(run + 1)
                dataset_results["Initial Density"].append(initial_density)
                dataset_results["LDP Density"].append(density_ldp)
                dataset_results["Baseline Density"].append(baseline_density)

                print(f"Dataset: {filename}, Run: {run+1}, ε: {epsilon}, LDP Density: {density_ldp:.4f}, Baseline Density: {baseline_density:.4f}")

        # 保存当前数据集的实验结果到CSV
        results_df = pd.DataFrame(dataset_results)
        results_csv_file = os.path.join(output_draw_dir, f"{dataset_name}_results.csv")
        results_df.to_csv(results_csv_file, index=False)

        print(f"Results for {filename} saved successfully to {results_csv_file}")