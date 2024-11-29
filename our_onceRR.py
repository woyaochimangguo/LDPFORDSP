import numpy as np
import networkx as nx
import random

def rr_noisy_vector(vector, epsilon, seed):
    """
    对输入的向量应用随机响应（RR）算法进行加噪，使用独立的随机种子。
    """
    random.seed(seed)  # 设置随机种子
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # RR概率
    noisy_vector = []
    for v in vector:
        if random.random() < p:
            noisy_vector.append(v)
        else:
            noisy_vector.append(1 - v)
    return np.array(noisy_vector), p

def generate_noisy_graph(G, epsilon):
    """
    基于RR算法对邻接矩阵的下三角部分加噪，生成一个噪声图。
    确保噪声图的节点集与原始图完全一致。
    """
    n = len(G.nodes)
    noisy_adj = np.zeros((n, n))
    nodes = list(G.nodes)

    for i in range(n):
        for j in range(i):  # 仅处理下三角部分
            seed = hash(f"{nodes[i]}-{nodes[j]}") % (2**32)  # 为每个节点对生成随机种子
            vector, p = rr_noisy_vector([G.has_edge(nodes[i], nodes[j])], epsilon, seed)
            noisy_adj[i, j] = vector[0]

    # 对称生成上三角部分
    noisy_adj = noisy_adj + noisy_adj.T

    # 构造噪声图
    noisy_G = nx.Graph()
    noisy_G.add_nodes_from(G.nodes)  # 确保噪声图的节点集与原始图一致
    for i in range(n):
        for j in range(i + 1, n):
            if noisy_adj[i, j] == 1:
                noisy_G.add_edge(nodes[i], nodes[j])

    return noisy_G, p

def estimate_degree(noisy_G, p):
    """
    计算噪声图中每个节点的估计度数。
    """
    n = len(noisy_G.nodes)
    estimated_degrees = {}
    for node in noisy_G.nodes:
        original_degree = noisy_G.degree[node]
        estimated_degree = original_degree * p + (n - 1 - original_degree) * (1 - p)
        estimated_degrees[node] = estimated_degree
    return estimated_degrees

def ldp_greedy_peeling(G, epsilon):
    """
    基于估计度数进行贪心剥离算法，返回最密子集。
    """
    noisy_G, p = generate_noisy_graph(G, epsilon)
    estimated_degrees = estimate_degree(noisy_G, p)
    max_density = 0
    densest_subset = set()
    subgraph = noisy_G.copy()

    # 贪心剥离过程
    while subgraph.number_of_nodes() > 0:
        total_estimated_degree = sum(estimated_degrees.values())
        current_density = total_estimated_degree / (2 * subgraph.number_of_nodes())  # 每条边贡献2，需除以2

        if current_density > max_density:
            max_density = current_density
            densest_subset = set(subgraph.nodes)

        min_node = min(estimated_degrees, key=estimated_degrees.get)
        subgraph.remove_node(min_node)
        del estimated_degrees[min_node]

        for node in subgraph.nodes:
            original_degree = subgraph.degree[node]
            estimated_degree = original_degree * p + (subgraph.number_of_nodes() - 1 - original_degree) * (1 - p)
            estimated_degrees[node] = estimated_degree

    return list(densest_subset)

def validate_subset_density_original(G, subset):
    """
    计算返回子集在原始图中的真实密度。
    """
    induced_subgraph = G.subgraph(subset)
    num_edges = induced_subgraph.number_of_edges()
    num_nodes = induced_subgraph.number_of_nodes()
    if num_nodes <= 1:
        return 0
    return  num_edges / num_nodes