import numpy as np
import networkx as nx
import hashlib

def jaccard_similarity(set1, set2):
    """计算 Jaccard 相似度"""
    intersection_size = len(set1 & set2)
    union_size = len(set1 | set2)
    return intersection_size / union_size if union_size > 0 else 0
def generate_seed(node_id):
    """生成基于节点 ID 的唯一随机种子"""
    hash_object = hashlib.md5(str(node_id).encode())
    return int(hash_object.hexdigest(), 16) % (2**32)

def randomized_response_vectorized(values, epsilon, rng):
    """向量化的随机响应机制：对多个二值元素进行随机响应处理"""
    p = np.exp(epsilon) / (1 + np.exp(epsilon))  # 保留概率
    random_values = rng.random(len(values)) < p  # 生成随机掩码
    return np.where(random_values, values, 1 - values)  # 应用随机响应

def add_dp_noise_to_vector(vector, epsilon, seed):
    """对节点的邻接向量进行随机响应加噪"""
    rng = np.random.default_rng(seed)  # 基于种子创建独立的随机数生成器
    return randomized_response_vectorized(vector, epsilon, rng)

def apply_rr_out_neighbors(G, epsilon):
    """对所有节点的出邻居列表应用随机响应保护"""
    perturbed_out_neighbors = {}

    for v in G.nodes():
        seed = generate_seed(v)  # 确保每个节点的随机数生成器不同
        out_neighbors = {u: 1 for u in G.successors(v)}
        out_vector = np.array([out_neighbors.get(u, 0) for u in G.nodes()])  # 转换为向量
        perturbed_vector = add_dp_noise_to_vector(out_vector, epsilon, seed)

        rr_neighbors = {u: perturbed_vector[i] for i, u in enumerate(G.nodes()) if perturbed_vector[i] == 1}
        perturbed_out_neighbors[v] = rr_neighbors

    return perturbed_out_neighbors

def build_perturbed_graph(perturbed_out_neighbors):
    """使用随机响应后的数据构造新的扰动图"""
    G_prime = nx.DiGraph()
    for v, neighbors in perturbed_out_neighbors.items():
        for u in neighbors:
            G_prime.add_edge(v, u)
    return G_prime

def calculate_density(G, S, T):
    if len(S) == 0 or len(T) == 0:
        return 0
    E_ST = [(u, v) for u in S for v in T if G.has_edge(u, v)]
    # 进行平方根标准化
    return len(E_ST) / (len(S) * len(T)) ** 0.5

#baseline1 without LDP
def densest_subgraph_directed(G):
    n = len(G.nodes())
    H = G.copy()
    max_density = 0
    max_subgraph = None
    max_S = H
    max_T = H

    # 计算原始图的密度
    initial_S = {node for node in G.nodes() if G.out_degree(node) > 0}  # 非零出度的节点
    initial_T = {node for node in G.nodes() if G.in_degree(node) > 0}   # 非零入度的节点
    initial_density = calculate_density(G, initial_S, initial_T)
    print(f"原始图的密度: {initial_density:.4f}")

    while H.number_of_nodes() > 0:
        # 计算每个顶点的入度和出度
        in_degrees = dict(H.in_degree())
        out_degrees = dict(H.out_degree())

        # 选择最小度的非零节点
        vi = min((node for node in H.nodes() if in_degrees[node] > 0), key=lambda v: in_degrees[v], default=None)
        vo = min((node for node in H.nodes() if out_degrees[node] > 0), key=lambda v: out_degrees[v], default=None)

        # 如果图中没有节点了，退出
        if vi is None or vo is None:
            break

        # 判断最小度节点的类别
        if in_degrees[vi] <= out_degrees[vo]:
            min_node = vi
            category = 'IN'
        else:
            min_node = vo
            category = 'OUT'

        # 输出选中的节点和其度数
        print(f"选择节点 {min_node}，入度: {in_degrees.get(min_node, 0)}，出度: {out_degrees.get(min_node, 0)}")

        # 删除度数最小的顶点及其相关边
        if category == 'IN':
            edges_to_remove = list(H.in_edges(min_node))
            H.remove_edges_from(edges_to_remove)
            print(f"删除节点 {min_node} 的入边: {edges_to_remove}")
        else:
            edges_to_remove = list(H.out_edges(min_node))
            H.remove_edges_from(edges_to_remove)
            print(f"删除节点 {min_node} 的出边: {edges_to_remove}")

        # 如果节点没有任何边，删除该节点
        if H.degree(min_node) == 0:
            H.remove_node(min_node)
            print(f"删除节点 {min_node}，因为它没有边与之相连")

        # 计算当前子图的密度
        S = {node for node in H.nodes() if H.out_degree(node) > 0}  # 非零出度的节点
        T = {node for node in H.nodes() if H.in_degree(node) > 0}   # 非零入度的节点
        current_density = calculate_density(H, S, T)
        print(f"当前子图的密度: {current_density:.4f}")

        # 更新最优密度和最优子图
        if current_density > max_density:
            max_density = current_density
            max_subgraph = H.copy()
            max_S = S.copy()
            max_T = T.copy()
            print(f"更新最优密度: {max_density:.4f}")

    return  max_S, max_T

def densest_subgraph_LDP(G, epsilon):
    """LDP-Densest-Subgraph-Directed 算法"""
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    perturbed_out_neighbors = apply_rr_out_neighbors(G, epsilon)
    G_prime = build_perturbed_graph(perturbed_out_neighbors)

    S, T = set(G_prime.nodes()), set(G_prime.nodes())
    rho_max = 0
    S_best, T_best = S, T

    # 计算初始的 dout_e 和 din_e
    dout_n = {v: G_prime.out_degree(v) for v in G_prime.nodes()}
    din_n = {v: G_prime.in_degree(v) for v in G_prime.nodes()}
    dout_e = {v: (dout_n[v] - len(G.nodes()) * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}
    din_e = {v: (din_n[v] - len(G.nodes()) * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}

    while S and T:
        # 选择度最小的节点
        u = min(S | T, key=lambda v: min(dout_e[v], din_e[v]))

        if dout_e[u] < din_e[u]:
            # 删除出边
            for v in list(G_prime.successors(u)):
                if (u, v) in G_prime.edges:
                    din_e[v] -= p / (2 * p - 1)
                else:
                    din_e[v] -= (p - 1) / (2 * p - 1)
            G_prime.remove_node(u)
        else:
            # 删除入边
            for v in list(G_prime.predecessors(u)):
                if (v, u) in G_prime.edges:
                    dout_e[v] -= p / (2 * p - 1)
                else:
                    dout_e[v] -= (p - 1) / (2 * p - 1)
            G_prime.remove_node(u)

        # 更新 S 和 T
        S = {v for v in G_prime.nodes() if dout_e[v] > 0}
        T = {v for v in G_prime.nodes() if din_e[v] > 0}

        # 计算密度
        E_ST = sum(1 for u in S for v in T if G_prime.has_edge(u, v))
        rho = (E_ST - (1 - p) * len(S) * len(T)) / ((2 * p - 1) * (len(S) * len(T)) ** 0.5)

        if rho > rho_max:
            rho_max = rho
            S_best, T_best = S.copy(), T.copy()

    return S_best, T_best






