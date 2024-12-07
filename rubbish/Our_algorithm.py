import numpy as np
import networkx as nx
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
import hashlib

def generate_seed(node_id):
    """生成基于节点 ID 的唯一随机种子"""
    hash_object = hashlib.md5(str(node_id).encode())
    return int(hash_object.hexdigest(), 16) % (2**32)

def randomized_response(value, epsilon):
    """随机响应机制：对单个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # 保留概率
    noise = np.random.rand() < p  # 以 p 的概率保留原值，以 1-p 的概率翻转
    return value if noise else 1 - value

def randomized_response_vectorized(values, epsilon):
    """向量化的随机响应机制：对多个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # 保留概率
    random_values = np.random.rand(len(values)) < p  # 生成随机掩码
    return np.where(random_values, values, 1 - values)  # 应用随机响应

def add_dp_noise_to_vector(vector, epsilon, seed):
    """对节点的邻接向量进行随机响应加噪"""
    rng = np.random.default_rng(seed)  # 为该节点创建固定随机数生成器
    np.random.seed(seed)  # 设置随机种子
    return randomized_response_vectorized(vector, epsilon)

def ldp_charikar_peeling_distributed(g, epsilon):
    """分布式 LDP 隐私保护与最密子图求解"""
    # 保存输入图的副本
    g_noisy = g.copy()  # 创建输入图的副本，用于加噪后的修改

    # 将图转为密集邻接矩阵
    adj_matrix = nx.to_numpy_array(g).astype(int)
    num_nodes = adj_matrix.shape[0]
    nodes = list(g.nodes)  # 保留原始节点标记

    # 第一次加噪：每个节点独立对完整邻接向量进行随机响应
    for node_id in range(num_nodes):
        # 获取节点的完整邻接向量
        neighbors = adj_matrix[node_id, :]  # 获取该节点的完整邻接向量

        # 生成随机种子并加噪
        seed = generate_seed(nodes[node_id])  # 使用原始节点标识生成种子
        noisy_vector = add_dp_noise_to_vector(neighbors,  epsilon, seed)  # 对完整向量加噪（epsilon 分摊）

        # 更新噪声矩阵的对应行
        adj_matrix[node_id, :] = noisy_vector  # 更新对应行


    # 对称化矩阵，确保上三角部分与下三角部分一致
    adj_matrix = adj_matrix + adj_matrix.T - np.diag(adj_matrix.diagonal())

    # 将加噪后的矩阵转换为图对象，确保节点与输入图保持一致
    noisy_graph = nx.from_numpy_array(adj_matrix)

    # 重命名节点以匹配原始图的节点标记
    mapping = {i: nodes[i] for i in range(num_nodes)}
    noisy_graph = nx.relabel_nodes(noisy_graph, mapping)

    # 验证节点一致性
    print("原图节点集合:", set(g.nodes))
    print("噪声图节点集合:", set(noisy_graph.nodes))
    assert set(g.nodes) == set(noisy_graph.nodes), "节点集合不一致！"

    # 运行贪婪算法求解最密子图
    dense_subgraph, density = charikar_peeling(noisy_graph)
    return dense_subgraph, density