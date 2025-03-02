import numpy as np
import networkx as nx
import hashlib
import os
from ReadGraph import  *

def generate_seed(node_id):
    """生成基于节点 ID 的唯一随机种子"""
    hash_object = hashlib.md5(str(node_id).encode())
    return int(hash_object.hexdigest(), 16) % (2**32)

def geometric_noise(epsilon):
    """
    Generate symmetric geometric noise with parameter epsilon.
    """
    p = np.exp(-epsilon)
    u = np.random.uniform(0, 1)
    noise = np.sign(u - 0.5) * np.floor(np.log(1 - 2 * abs(u - 0.5)) / np.log(p))
    return int(noise)

def randomized_response_vectorized(values, epsilon, rng):
    """向量化的随机响应机制：对多个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # 保留概率
    random_values = rng.random(len(values)) < p  # 使用指定随机数生成器生成随机掩码
    return np.where(random_values, values, 1 - values)  # 应用随机响应

def add_dp_noise_to_vector(vector, epsilon, seed):
    """对节点的邻接向量进行随机响应加噪"""
    rng = np.random.default_rng(seed)  # 基于种子创建独立的随机数生成器
    return randomized_response_vectorized(vector, epsilon, rng)

def correct_noise(Ri, dr, dg):
    """
    基于噪声度数 dr 和 dg 来纠正噪声，调整邻接矩阵中的噪声。
    只操作下三角部分
    """
    len_Ri = len(Ri)  # 邻接向量的长度

    # 如果 dg 小于等于 0，将所有邻接向量设置为 0
    if dg <= 0:
        Ri = np.zeros(len_Ri, dtype=int)
    # 如果 dg 大于等于邻接向量长度，将所有邻接向量设置为 1
    elif dg >= len_Ri:
        Ri = np.ones(len_Ri, dtype=int)
    else:
        # 如果 dg <= dr，翻转1为0的概率为 dg/dr
        if dg <= dr:
            p_flip_1 = dg / dr if dr > 0 else 0  # 防止除零错误
            # 以 p_flip_1 的概率翻转所有1为0
            for j in range(len_Ri):
                if Ri[j] == 1:
                    if np.random.random() < p_flip_1:
                        Ri[j] = 0
        # 如果 dg > dr，翻转0为1的概率为 (dg - dr) / (len(Ri) - dr)
        else:
            p_flip_0 = (dg - dr) / (len_Ri - dr) if len_Ri - dr > 0 else 0  # 防止除零错误
            # 以 p_flip_0 的概率翻转所有0为1
            for j in range(len_Ri):
                if Ri[j] == 0:
                    if np.random.random() < p_flip_0:
                        Ri[j] = 1

    return Ri

def generate_noisy_graph(g, epsilon1, epsilon2):
    """
    基于输入图 G 和隐私预算 epsilon，为每个节点生成独立种子并加噪，最终仅使用下三角部分生成噪声图。
    """
    # 将图转换为邻接矩阵
    adj_matrix = nx.to_numpy_array(g).astype(int)
    num_nodes = adj_matrix.shape[0]
    nodes = list(g.nodes)  # 节点列表

    # 初始化噪声矩阵
    noisy_matrix = np.zeros_like(adj_matrix, dtype=int)

    # 对每个节点生成独立种子并对其邻接向量加噪
    for i in range(num_nodes):
        neighbors = adj_matrix[i, :]  # 节点 i 的邻接向量
        seed = generate_seed(nodes[i])  # 基于节点 i 生成独立种子
        noisy_neighbors = add_dp_noise_to_vector(neighbors, epsilon1, seed)  # 对邻接向量加噪

        # 获取原始图的下三角部分度数
        true_degree = sum(adj_matrix[i, :i])  # 原始图的下三角部分度数
        dg = true_degree + geometric_noise(epsilon2)  # 加几何噪声后的度数

        # 只对下三角部分进行处理
        noisy_neighbors = noisy_neighbors[:i]  # 只处理下三角部分

        # 计算噪声度数 dr 和几何噪声 dg
        dr = sum(noisy_neighbors)  # 随机响应后的度数

        # 纠正噪声
        noisy_neighbors = correct_noise(noisy_neighbors, dr, dg)

        noisy_matrix[i, :i] = noisy_neighbors  # 更新噪声矩阵的下三角部分

    # 提取下三角部分生成噪声图
    lower_triangle = np.tril(noisy_matrix, k=-1)  # 仅保留下三角部分
    noisy_graph = nx.from_numpy_array(lower_triangle)  # 从下三角矩阵生成无向图

    # 恢复节点标记为原始图的节点标记
    mapping = {i: nodes[i] for i in range(num_nodes)}
    noisy_graph = nx.relabel_nodes(noisy_graph, mapping)

    return noisy_graph

def validate_subset_density_original(G, subset):
    """
    在原始图中计算子集的真实密度。
    """
    induced_subgraph = G.subgraph(subset)
    num_edges = induced_subgraph.number_of_edges()
    num_nodes = induced_subgraph.number_of_nodes()
    if num_nodes <= 1:
        return 0
    return num_edges / num_nodes

def greedy_algorithm_with_algorithm2(g, epsilon1, epsilon2):
    """
    基于差分隐私的贪心算法，返回最稠密子图。
    """
    # 生成噪声图
    noisy_graph = generate_noisy_graph(g, epsilon1, epsilon2)

    # 初始化
    max_density = 0
    max_subgraph = []
    nodes = list(noisy_graph.nodes)

    # 创建一个初步包含所有节点的集合 S
    S = nodes[:]

    # 计算初始的密度
    density = validate_subset_density_original(noisy_graph, S)
    max_density = density
    max_subgraph = S

    # 从图中选择一个节点并逐步剔除，直到剩下1个节点
    while len(S) > 1:
        # 找到当前子图 S 中度最小的节点
        vmin = min(S, key=lambda v: len(list(noisy_graph.neighbors(v))))
        S.remove(vmin)

        # 计算当前子图的密度
        density = validate_subset_density_original(noisy_graph, S)
        print("这一轮的度数为", density)

        # 更新最稠密子图
        if density > max_density:
            max_density = density
            max_subgraph = S[:]

    return max_subgraph

