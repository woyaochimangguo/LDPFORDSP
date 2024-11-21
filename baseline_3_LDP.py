import numpy as np
import networkx as nx
from ReadGraph import *

def ledp_densest_subgraph(graph, epsilon, log_n, rounds=10):
    """
    基于 LEDP 的最密子图计算，支持 NetworkX 图对象。

    参数：
    - graph: NetworkX 图对象
    - epsilon: 隐私预算
    - log_n: 图中节点数量的对数
    - rounds: 迭代轮次（用于 k-core 分解）

    返回：
    - densest_subgraph: 最密子图的节点集
    - max_density: 最密子图的密度
    """
    # 初始化
    core_numbers = {v: 0 for v in graph.nodes}
    lap_scale = 4 / epsilon  # 拉普拉斯噪声尺度

    # 初始化每个节点的噪声阈值
    noise_thresholds = {v: np.random.laplace(scale=lap_scale) for v in graph.nodes}

    max_density = 0
    densest_subgraph = set()

    for k in range(1, rounds + 1):
        next_graph = nx.Graph()  # 临时存储下一轮的子图
        for node in graph.nodes:
            noisy_degree = graph.degree[node] + np.random.laplace(scale=lap_scale)
            if noisy_degree >= k + noise_thresholds[node]:
                next_graph.add_node(node)  # 节点保留
                for neighbor in graph.neighbors(node):
                    if neighbor in next_graph:
                        next_graph.add_edge(node, neighbor)  # 保留边

        # 计算当前子图的密度
        if next_graph.number_of_nodes() > 0:
            subgraph_edges = next_graph.number_of_edges()
            subgraph_density = subgraph_edges / next_graph.number_of_nodes()
            if subgraph_density > max_density:
                max_density = subgraph_density
                densest_subgraph = set(next_graph.nodes)

        graph = next_graph  # 更新图结构
        if graph.number_of_nodes() == 0:
            break  # 如果没有剩余节点，结束迭代

    return densest_subgraph, max_density

# 示例调用
prefix = './datasets/Facebook/facebook/'
files = ['414.edges', '107.edges']
epsilon = 3
delta = 1e-9
# 读取文件并添加边
G = readFacebook(prefix, files[1])
log_n = np.log(G.number_of_nodes())

densest_subgraph, max_density = ledp_densest_subgraph(G, epsilon, log_n)
print("最密子图的节点集:", densest_subgraph)
print("最密子图的密度:", max_density)