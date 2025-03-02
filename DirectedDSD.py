import gzip
import pandas as pd
import networkx as nx
from DirectedGraphBaseline import *
from DirectedGraphOurMethod import *

# 读取数据集
def load_bitcoinotc_dataset(file_path):
    with gzip.open(file_path, 'rt') as file:
        data = pd.read_csv(file, header=None)
    data.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
    return data

# 构建有向图，忽略边的权重，设置为1
def build_directed_graph(data):
    G = nx.DiGraph()
    for _, row in data.iterrows():
        # 设置边的权重为1
        G.add_edge(row['SOURCE'], row['TARGET'], weight=1)
    return G

# 密度计算（进行平方根标准化）
def calculate_density(G, S, T):
    if len(S) == 0 or len(T) == 0:
        return 0
    E_ST = [(u, v) for u in S for v in T if G.has_edge(u, v)]
    # 进行平方根标准化
    return len(E_ST) / (len(S) * len(T)) ** 0.5

# Jaccard相似度计算
def jaccard_similarity(S, T):
    intersection = len(S & T)
    union = len(S | T)
    return intersection / union if union != 0 else 0

# 密集子图算法（Densest Subgraph for Directed Graphs）


def build_directed_graph_limit(data, node_limit=200):
    G = nx.DiGraph()
    nodes = set()
    # 添加边并记录节点
    for _, row in data.iterrows():
        source = row['SOURCE']
        target = row['TARGET']
        if source not in nodes and len(nodes) < node_limit:
            nodes.add(source)
        if target not in nodes and len(nodes) < node_limit:
            nodes.add(target)
        if source in nodes and target in nodes:
            G.add_edge(source, target, weight=1)  # 设置权重为1
        if len(nodes) >= node_limit:
            break
    return G

# 主函数
if __name__ == "__main__":
    file_path = 'DirectedDatasets/soc-sign-bitcoinotc.csv.gz'
    data = load_bitcoinotc_dataset(file_path)
    G = build_directed_graph_limit(data,200)

    print("图构建完成，节点数:", G.number_of_nodes(), "边数:", G.number_of_edges())

    max_subgraph, max_density, max_S, max_T, initial_density = densest_subgraph_LDP(G)

    print("最密集子图的节点数:", max_subgraph.number_of_nodes())
    print("最密集子图的边数:", max_subgraph.number_of_edges())
    print("最密集子图的密度:", max_density)
    print("最密集子图的节点集合 S:", max_S)
    print("最密集子图的节点集合 T:", max_T)

    # 计算S和T的Jaccard相似度
    similarity = jaccard_similarity(max_S, max_T)
    print(f"最终集合 S 和 T 的 Jaccard 相似度: {similarity:.4f}")

    # 输出原始图的密度
    print(f"原始图的密度: {initial_density:.4f}")