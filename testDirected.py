import networkx as nx
import pandas as pd
import gzip
from DirectedGraphOurMethod import *

def load_bitcoinotc_dataset(file_path):
    """加载 Bitcoin OTC 数据集"""
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, header=None, usecols=[0, 1], names=['source', 'target'])
    return df

def build_directed_graph(data):
    """构建完整的有向图（不限制节点数）"""
    G = nx.DiGraph()
    for _, row in data.iterrows():
        src, tgt = int(row['source']), int(row['target'])
        G.add_edge(src, tgt)
    return G

def calculate_density(G, S, T):
    """计算 S 和 T 之间的密度"""
    if len(S) == 0 or len(T) == 0:
        return 0
    E_ST = [(u, v) for u in S for v in T if G.has_edge(u, v)]
    return len(E_ST) / (len(S) * len(T)) ** 0.5

def jaccard_similarity(set1, set2):
    """计算 Jaccard 相似度"""
    intersection_size = len(set1 & set2)
    union_size = len(set1 | set2)
    return intersection_size / union_size if union_size > 0 else 0

def densest_subgraph_directed(G):
    """Baseline1: 无隐私保护的最密子图算法"""
    H = G.copy()
    max_density = 0
    max_subgraph = None
    max_S = H
    max_T = H

    initial_S = {node for node in G.nodes() if G.out_degree(node) > 0}
    initial_T = {node for node in G.nodes() if G.in_degree(node) > 0}
    initial_density = calculate_density(G, initial_S, initial_T)

    while H.number_of_nodes() > 0:
        in_degrees = dict(H.in_degree())
        out_degrees = dict(H.out_degree())

        vi = min((node for node in H.nodes() if in_degrees[node] > 0), key=lambda v: in_degrees[v], default=None)
        vo = min((node for node in H.nodes() if out_degrees[node] > 0), key=lambda v: out_degrees[v], default=None)

        if vi is None or vo is None:
            break

        if in_degrees[vi] <= out_degrees[vo]:
            min_node = vi
        else:
            min_node = vo

        H.remove_node(min_node)

        S = {node for node in H.nodes() if H.out_degree(node) > 0}
        T = {node for node in H.nodes() if H.in_degree(node) > 0}
        current_density = calculate_density(H, S, T)

        if current_density > max_density:
            max_density = current_density
            max_subgraph = H.copy()
            max_S = S.copy()
            max_T = T.copy()

    return max_subgraph, max_density, max_S, max_T, initial_density

# 1. **加载完整数据集**
file_path = 'DirectedDatasets/soc-sign-bitcoinotc.csv.gz'
data = load_bitcoinotc_dataset(file_path)

# 2. **构建完整有向图**
G = build_directed_graph(data)

# 3. **Baseline1: 计算原始最密集子图**
max_subgraph_baseline, max_density_baseline, S_baseline, T_baseline, initial_density_baseline = densest_subgraph_directed(G)

print("\n===== Baseline1 结果 =====")
print(f"Baseline1 计算出的最密集子图密度（在原始图上）: {max_density_baseline:.4f}")

# 4. **不同 ε 进行 LDP 计算**
epsilons = [1, 2, 3, 4, 5, 6, 7, 9]
results = []

for epsilon in epsilons:
    S_LDP, T_LDP, _ = densest_subgraph_LDP(G, epsilon)  # LDP 计算不需要 LDP 图的密度

    # 在原始图上计算 S_LDP 和 T_LDP 的密度
    original_density_LDP = calculate_density(G, S_LDP, T_LDP)

    # 计算集合相似度
    sim_S = jaccard_similarity(S_LDP, S_baseline)
    sim_T = jaccard_similarity(T_LDP, T_baseline)

    results.append((epsilon, original_density_LDP, sim_S, sim_T))

    print(f"\n===== ε = {epsilon} =====")
    print(f"在原始图上计算的 LDP 子图密度: {original_density_LDP:.4f}")
    print(f"S 的 Jaccard 相似度: {sim_S:.4f}")
    print(f"T 的 Jaccard 相似度: {sim_T:.4f}")

# 5. **最终结果对比**
print("\n===== 结果对比表 =====")
print(f"{'ε':<5} {'Baseline密度':<12} {'LDP原始密度':<12} {'S相似度':<10} {'T相似度'}")
for epsilon, orig_density, sim_S, sim_T in results:
    print(f"{epsilon:<5} {max_density_baseline:<12.4f} {orig_density:<12.4f} {sim_S:<10.4f} {sim_T:.4f}")