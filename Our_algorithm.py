import numpy as np
from scipy.sparse import coo_matrix, tril
import networkx as nx
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
def randomized_response(value, epsilon):
    """随机响应机制：对单个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    noise = np.random.rand() < p  # 以 p 的概率保留原值，以 1-p 的概率反转
    return value if noise else 1 - value

def add_dp_noise_to_sparse_matrix(sparse_matrix, epsilon):
    """对稀疏矩阵的下三角部分应用随机响应保护"""
    # 获取严格下三角部分的稀疏矩阵
    lower_tri = tril(sparse_matrix, k=-1).tocoo()

    # 对每个非零元素应用随机响应
    noisy_data = [
        randomized_response(value, epsilon) for value in lower_tri.data
    ]

    # 构造具有噪声的稀疏矩阵
    noisy_lower_tri = coo_matrix(
        (noisy_data, (lower_tri.row, lower_tri.col)),
        shape=lower_tri.shape
    )

    # 将下三角矩阵对称化，确保结果仍是无向图
    return noisy_lower_tri + noisy_lower_tri.T

def ldp_charikar_peeling(g, epsilon):
    # 采用稀疏矩阵的方式存储图
    sparse_adj_matrix = nx.to_scipy_sparse_array(g, format='coo')

    # 进行第一次翻转
    dp_protected_matrix = add_dp_noise_to_sparse_matrix(sparse_adj_matrix, epsilon)

    # 进行第二次翻转
    ldp_protected_matrix = add_dp_noise_to_sparse_matrix(dp_protected_matrix, epsilon)
    ldp_protected_matrix = ldp_protected_matrix.tocoo()
    # 生成翻转以后的噪声图
    # 生成噪声图，保留原始节点标记
    ldp_noisy_graph = nx.Graph()

    # 获取原图节点的顺序，用于映射稀疏矩阵索引
    node_mapping = list(g.nodes)

    # 遍历加噪后的稀疏矩阵并添加边，同时保留原始节点标记
    for u, v, d in zip(ldp_protected_matrix.row, ldp_protected_matrix.col, ldp_protected_matrix.data):
        if d > 0:  # 仅保留权重大于 0 的边
            original_u = node_mapping[u]
            original_v = node_mapping[v]
            ldp_noisy_graph.add_edge(original_u, original_v)
    #ldp_noisy_graph = nx.from_scipy_sparse_array(ldp_protected_matrix)

    dense_subgraph, density = charikar_peeling(ldp_noisy_graph)
    return  dense_subgraph,density