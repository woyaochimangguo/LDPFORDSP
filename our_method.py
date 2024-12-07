import numpy as np
import hashlib
from ReadGraph import *
from baseline_1_GreedyPeeling_withoutDP import *

def generate_seed(node_id):
    """生成基于节点 ID 的唯一随机种子"""
    hash_object = hashlib.md5(str(node_id).encode())
    return int(hash_object.hexdigest(), 16) % (2**32)

def randomized_response_vectorized(values, epsilon, rng):
    """向量化的随机响应机制：对多个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # 保留概率
    random_values = rng.random(len(values)) < p  # 使用指定随机数生成器生成随机掩码
    return np.where(random_values, values, 1 - values)  # 应用随机响应

def add_dp_noise_to_vector(vector, epsilon, seed):
    """对节点的邻接向量进行随机响应加噪"""
    rng = np.random.default_rng(seed)  # 基于种子创建独立的随机数生成器
    return randomized_response_vectorized(vector, epsilon, rng)

def generate_noisy_graph(g, epsilon):
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
        noisy_neighbors = add_dp_noise_to_vector(neighbors, epsilon, seed)  # 对邻接向量加噪
        noisy_matrix[i, :] = noisy_neighbors  # 更新噪声矩阵

    # 提取下三角部分生成噪声图
    lower_triangle = np.tril(noisy_matrix, k=-1)  # 仅保留下三角部分
    noisy_graph = nx.from_numpy_array(lower_triangle)  # 从下三角矩阵生成无向图

    # 恢复节点标记为原始图的节点标记
    mapping = {i: nodes[i] for i in range(num_nodes)}
    noisy_graph = nx.relabel_nodes(noisy_graph, mapping)

    return noisy_graph

def ldp_greedy_peeling(g, epsilon):
    """
    使用估计度数的贪心剥离算法计算最密子集，基于噪声图的度数进行纠偏。
    """
    realdegree = {node: g.degree[node] for node in g.nodes}
    print("Real Degrees:", realdegree)

    noisy_G = generate_noisy_graph(g, epsilon)
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    n = len(noisy_G.nodes)

    estimated_degrees = {}
    for node in noisy_G.nodes:
        x = noisy_G.degree[node]
        d = (x + n * p - n) / (2 * p - 1)
        estimated_degrees[node] = d
    print("Initial estimated Degrees:", estimated_degrees)
    print("-" * 50)

    max_density = 0
    densest_subset = set()
    subgraph = noisy_G.copy()

    iteration = 0
    while subgraph.number_of_nodes() > 0:
        iteration += 1
        total_estimated_degree = sum(estimated_degrees.values())
        current_density = total_estimated_degree / (2 * subgraph.number_of_nodes())

        if current_density > max_density:
            max_density = current_density
            densest_subset = set(subgraph.nodes)

        min_node = min(estimated_degrees, key=estimated_degrees.get)
        print(f"Iteration {iteration}:")
        print(f"  Current estimated density: {current_density:.4f}")
        print(f"  Removing node {min_node} with estimated degree: {estimated_degrees[min_node]:.4f}")
        print("-" * 50)

        # 更新剩余节点的估计度数
        for node in subgraph.nodes:
            d_original = estimated_degrees[node]
            if min_node in subgraph.neighbors(node):
                d_updated = d_original - (p / (2 * p - 1))
            else:
                d_updated = d_original - ((p - 1) / (2 * p - 1))
            estimated_degrees[node] = d_updated

        subgraph.remove_node(min_node)
        del estimated_degrees[min_node]

    return list(densest_subset)

def ldp_greedy_peeling_withoutestimate(g, epsilon):
    noisy_G = generate_noisy_graph(g, epsilon)
    de ,ds = charikar_peeling(noisy_G)
    return de,ds


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
