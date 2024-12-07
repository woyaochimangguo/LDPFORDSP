import numpy as np
import networkx as nx
from ReadGraph import *
def laplace_noise(scale):
    """
    生成拉普拉斯噪声
    :param scale: 拉普拉斯噪声的尺度参数
    :return: 噪声值
    """
    return np.random.laplace(0, scale)

def k_core_decomposition(graph, epsilon):
    """
    差分隐私的 k 核分解
    :param graph: NetworkX 图对象
    :param epsilon: 隐私参数
    :return: 节点的近似 k 核值字典
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # 初始化
    Vt = set(nodes)
    k = 60 * np.log(n) / epsilon
    k_core_values = {v: 0 for v in nodes}
    noise_l = {v: laplace_noise(4 / epsilon) for v in nodes}  # 初始化每个节点的噪声

    while k <= n:
        # 每次更新节点的集合
        Vt_prev = set(Vt)
        while True:
            to_remove = []
            for v in Vt:
                degree = sum(1 for neighbor in graph.neighbors(v) if neighbor in Vt_prev)
                noise_degree = degree + laplace_noise(8 / epsilon)  # 添加噪声到度数
                if noise_degree <= k + noise_l[v]:
                    to_remove.append(v)

            for v in to_remove:
                Vt.remove(v)

            if not to_remove:
                break

        # 更新 k 核值
        for v in Vt:
            k_core_values[v] = max(k_core_values[v], k)

        k += 60 * np.log(n) / epsilon

    return k_core_values

def densest_subgraph(graph, k_core_values, epsilon):
    """
    基于 k 核分解的最密子图
    :param graph: NetworkX 图对象
    :param k_core_values: 每个节点的近似 k 核值
    :param epsilon: 隐私参数
    :return: 最密子图的节点集合和密度
    """
    n = len(graph.nodes)
    c_prime = 10  # 常数因子，可根据实际需要调整

    # 找到最大核值
    k_max = max(k_core_values.values())
    threshold = k_max - c_prime * np.log(n) / epsilon

    # 找到满足条件的节点集合
    densest_nodes = [v for v, k in k_core_values.items() if k >= threshold]

    # 构造诱导子图
    densest_subgraph = graph.subgraph(densest_nodes)

    # 计算密度
    subgraph_edges = densest_subgraph.number_of_edges()
    subgraph_nodes = len(densest_nodes)
    density = subgraph_edges / subgraph_nodes if subgraph_nodes > 0 else 0

    return densest_nodes, density

# 主函数
if __name__ == "__main__":
    # 示例图
    prefix = './datasets/Facebook/facebook/'
    files = ['414.edges']
    graph = readFacebook(prefix, files[0])
    epsilon = 2.0  # 隐私预算

    print("执行差分隐私的 k 核分解...")
    k_core_values = k_core_decomposition(graph, epsilon)
    print("每个节点的近似 k 核值:", k_core_values)

    print("\n计算最密子图...")
    densest_nodes, density = densest_subgraph(graph, k_core_values, epsilon)
    print("最密子图的节点集合:", densest_nodes)
    print("最密子图的密度:", density)