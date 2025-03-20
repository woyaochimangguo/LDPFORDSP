import os
import time
from DirectedGraphBaseline import *
from ReadDirectedGraph import *

# 数据集路径和输出路径设置
datasets_dir = '/Users/teco/Documents/GitHub/LDPFORDSP/DirectedDatasets/datasets/'
output_base_dir = "result_txt"
os.makedirs(output_base_dir, exist_ok=True)

# 隐私预算设置
epsilons = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
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

        # 收集当前数据集的实验结果
        dataset_results = []

        # 针对每个隐私预算进行实验
        for epsilon in epsilons:
            densities, sims_S, sims_T, runtimes = [], [], [], []

            for _ in range(repeat_times):
                start_time = time.time()
                S_LDP, T_LDP = densest_subgraph_LDP(G, epsilon)
                end_time = time.time()

                density_ldp = calculate_density(G, S_LDP, T_LDP)
                sim_S = jaccard_similarity(S_LDP, S_baseline)
                sim_T = jaccard_similarity(T_LDP, T_baseline)
                runtime = end_time - start_time

                densities.append(density_ldp)
                sims_S.append(sim_S)
                sims_T.append(sim_T)
                runtimes.append(runtime)

            # 求五次实验的平均值
            avg_density = sum(densities) / repeat_times
            avg_sim_S = sum(sims_S) / repeat_times
            avg_sim_T = sum(sims_T) / repeat_times
            avg_runtime = sum(runtimes) / repeat_times

            # 记录结果
            dataset_results.append({
                "Epsilon": epsilon,
                "Avg LDP Density": avg_density,
                "Avg S Similarity": avg_sim_S,
                "Avg T Similarity": avg_sim_T,
                "Avg LDP Runtime": avg_runtime,
                "Baseline Density": baseline_density
            })

            print(f"Dataset: {filename}, ε: {epsilon}, Avg Density: {avg_density:.4f}, "
                  f"S Sim: {avg_sim_S:.4f}, T Sim: {avg_sim_T:.4f}, Runtime: {avg_runtime:.4f}s")

        # 保存当前数据集的实验结果到CSV
        results_df = pd.DataFrame(dataset_results)
        results_csv_file = os.path.join(output_draw_dir, f"{dataset_name}_results.csv")
        results_df.to_csv(results_csv_file, index=False)

        print(f"Results for {filename} saved successfully to {results_csv_file}")
