import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import hashlib
from ReadGraph import *

def generate_seed(node_id):
    """生成基于节点 ID 的唯一随机种子"""
    hash_object = hashlib.md5(str(node_id).encode())
    return int(hash_object.hexdigest(), 16) % (2**32)

def randomized_response_vectorized(values, epsilon):
    """向量化的随机响应机制：对多个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # 保留概率
    random_values = np.random.rand(len(values)) < p  # 生成随机掩码
    return np.where(random_values, values, 1 - values)  # 应用随机响应

def add_dp_noise_to_vector(vector, epsilon, seed):
    """对节点的邻接向量进行随机响应加噪"""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    return randomized_response_vectorized(vector, epsilon)

def generate_noisy_graph_lower_triangle(g, epsilon):
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
        noisy_neighbors = randomized_response_vectorized(neighbors, epsilon)  # 对邻接向量加噪
        noisy_matrix[i, :] = noisy_neighbors  # 更新噪声矩阵

    # 提取下三角部分生成噪声图
    lower_triangle = np.tril(noisy_matrix, k=-1)  # 仅保留下三角部分
    noisy_graph = nx.from_numpy_array(lower_triangle)  # 从下三角矩阵生成无向图

    # 恢复节点标记为原始图的节点标记
    mapping = {i: nodes[i] for i in range(num_nodes)}
    noisy_graph = nx.relabel_nodes(noisy_graph, mapping)

    return noisy_graph

def plot_distributions_with_lower_triangle(G, epsilons):
    """绘制多个隐私预算值下真实度数、下三角噪声图的度数、纠偏后的度数分布对比图。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, epsilon in enumerate(epsilons):
        realdegree = {node: G.degree[node] for node in G.nodes}
        noisy_G_lower = generate_noisy_graph_lower_triangle(G, epsilon)
        noisy_degree_lower = {node: noisy_G_lower.degree[node] for node in noisy_G_lower.nodes}

        p = np.exp(epsilon) / (1 + np.exp(epsilon))
        n = len(noisy_G_lower.nodes)
        estimated_degrees_lower = {
            node: (noisy_G_lower.degree[node] + n * p - n) / (2 * p - 1)
            for node in noisy_G_lower.nodes
        }

        ax = axes[i]
        ax.hist(realdegree.values(), bins=20, alpha=0.5, label="Real Degrees", edgecolor="black")
        ax.hist(noisy_degree_lower.values(), bins=20, alpha=0.5, label="Noisy Degrees (Lower)", edgecolor="black")
        ax.hist(
            estimated_degrees_lower.values(),
            bins=20,
            alpha=0.5,
            label="Estimated Degrees (Lower)",
            edgecolor="black",
        )
        ax.set_title(f"Epsilon = {epsilon}")
        ax.set_xlabel("Degree Values")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prefix = './datasets/Facebook/facebook/'
    files = ['414.edges', '107.edges', "0.edges", "348.edges", "686.edges", "698.edges", "1684.edges", "1912.edges", "3437.edges", "3980.edges"]

    G = nx.Graph()
    for file in files:
        G_ = readFacebook(prefix, file)
        G = nx.compose(G, G_)

    epsilons = [1, 2, 3, 4, 5, 8]
    plot_distributions_with_lower_triangle(G, epsilons)