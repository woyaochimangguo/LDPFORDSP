import numpy as np
import random


# 几何噪声生成（用于差分隐私）
def generate_noise(scale):
    """生成高斯噪声"""
    return np.random.normal(0, scale)


# Noisy-Order-Packing-MWU 实现
def noisy_order_packing_mwu(adj_matrix, epsilon, T, noise_scale):
    """
    Noisy Order Packing MWU
    :param adj_matrix: 图的邻接矩阵
    :param epsilon: 隐私预算
    :param T: 迭代轮数
    :param noise_scale: 噪声的尺度
    :return: 权重分布和节点排序
    """
    n = adj_matrix.shape[0]
    weights = np.ones(n)  # 初始化权重

    for t in range(T):
        # 计算概率分布
        probabilities = weights / np.sum(weights)

        # 根据权重计算排序
        ordering = np.argsort(-probabilities)

        # 计算带噪声的损失
        noisy_losses = []
        for i in range(n):
            subgraph_nodes = ordering[:i + 1]
            subgraph_edges = adj_matrix[subgraph_nodes][:, subgraph_nodes].sum() / 2
            density = subgraph_edges / (i + 1)
            noise = generate_noise(noise_scale)
            noisy_losses.append(-density + noise)  # 损失为负密度加噪声

        # 更新权重
        for i in range(n):
            weights[i] *= np.exp(-epsilon * noisy_losses[i])

    return weights, ordering


# DSG-LEDP-core 实现
def dsg_ledp_core(adj_matrix, epsilon, T, noise_scale):
    """
    差分隐私密子图核心算法
    :param adj_matrix: 图的邻接矩阵
    :param epsilon: 隐私预算
    :param T: 迭代轮数
    :param noise_scale: 噪声尺度
    :return: 密子图及其密度
    """
    n = adj_matrix.shape[0]
    loads = np.zeros(n)  # 初始化负载

    for t in range(T):
        # 根据负载排序节点
        ordering = np.argsort(-loads)

        # 计算排序中每个节点的带噪度
        noisy_degrees = []
        for i in range(n):
            subgraph_nodes = ordering[:i + 1]
            subgraph_edges = adj_matrix[subgraph_nodes][:, subgraph_nodes].sum() / 2
            noise = generate_noise(noise_scale)
            noisy_degrees.append(subgraph_edges + noise)

        # 更新负载
        for i in range(n):
            loads[i] += noisy_degrees[i]

    # 提取最优密子图
    best_density = 0
    best_subgraph = []
    for i in range(1, n + 1):
        subgraph = ordering[:i]
        density = adj_matrix[subgraph][:, subgraph].sum() / (2 * len(subgraph))
        if density > best_density:
            best_density = density
            best_subgraph = subgraph

    return best_subgraph, best_density


# 图数据加载函数
def load_graph(filepath):
    """
    从文件加载图数据
    :param filepath: 文件路径
    :return: 图的邻接矩阵
    """
    edges = []
    with open(filepath, 'r') as file:
        for line in file:
            u, v = map(int, line.strip().split())
            edges.append((u, v))

    # 获取节点数
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    n = max(nodes) + 1

    # 构造邻接矩阵
    adj_matrix = np.zeros((n, n))
    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    return adj_matrix


# 主函数
if __name__ == "__main__":
    # 参数设置
    filepath = "./datasets/Facebook/facebook/414.edges"
    epsilon = 3.0  # 隐私预算
    T = 10  # 迭代轮数
    noise_scale = 0.5  # 噪声尺度

    # 加载图数据
    print("加载图数据...")
    adj_matrix = load_graph(filepath)

    # 运行 DSG-LEDP-core 算法
    print("运行 DSG-LEDP-core 算法...")
    subgraph, density = dsg_ledp_core(adj_matrix, epsilon, T, noise_scale)
    print(f"最密子图节点集合: {subgraph}")
    print(f"估计密度: {density}")

    # 运行 Noisy-Order-Packing-MWU 算法
    print("运行 Noisy-Order-Packing-MWU 算法...")
    weights, ordering = noisy_order_packing_mwu(adj_matrix, epsilon, T, noise_scale)
    print(f"权重分布: {weights}")
    print(f"节点排序: {ordering}")