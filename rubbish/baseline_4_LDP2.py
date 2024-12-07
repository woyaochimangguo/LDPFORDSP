import numpy as np
import random
import networkx as nx
#paper Differential Privacy from Locally Adjustable Graph Algorithms:
#k-Core Decomposition, Low Out-Degree Ordering, and Densest
#Subgraphs
# 几何噪声生成函数
def geometric_noise(epsilon):
    """生成对称几何噪声"""
    if epsilon <= 0:
        raise ValueError("隐私预算 epsilon 必须大于 0")
    u = random.uniform(0, 1)
    sign = -1 if random.uniform(0, 1) < 0.5 else 1
    return sign * int(np.floor(np.log(1 - u) / np.log(1 - np.exp(-epsilon))))

# LEDP最密子图算法
def ledp_densest_subgraph(adj_list, epsilon, eta):
    """
    LEDP实现的最密子图算法
    :param adj_list: 图的邻接表表示
    :param epsilon: 隐私预算
    :param eta: 增长因子
    :return: 最密子图的节点集合及其密度
    """
    n = len(adj_list)
    max_density = 0
    best_subgraph = set()

    # 计算最大轮次
    max_iterations = int(np.log(n) / np.log(1 + eta))

    # 多轮迭代筛选子图
    for i in range(max_iterations):
        z = (1 + eta) ** i  # 当前密度阈值
        current_subgraph = set()

        # 计算带噪邻接边数
        noisy_degrees = {}
        for node, neighbors in adj_list.items():
            true_degree = len(neighbors)
            noisy_degree = true_degree + geometric_noise(epsilon / (6 * max_iterations))
            noisy_degrees[node] = noisy_degree

        # 根据密度阈值筛选节点
        for node, noisy_degree in noisy_degrees.items():
            if noisy_degree >= z:
                current_subgraph.add(node)

        # 计算当前子图密度
        current_edges = sum(len(adj_list[node]) for node in current_subgraph) / 2
        current_density = current_edges / max(1, len(current_subgraph))

        # 更新最优子图和密度
        if current_density > max_density:
            max_density = current_density
            best_subgraph = current_subgraph


    return best_subgraph, max_density

# 从文件读取图
def read_facebook_graph(prefix, filename):
    """
    从文件中读取Facebook图数据
    :param prefix: 文件路径前缀
    :param filename: 文件名
    :return: 邻接表表示的图
    """
    filepath = prefix + filename
    adj_list = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                u, v = map(int, line.strip().split())
                if u not in adj_list:
                    adj_list[u] = []
                if v not in adj_list:
                    adj_list[v] = []
                adj_list[u].append(v)
                adj_list[v].append(u)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {filepath} 不存在，请检查路径")
    return adj_list

# 实验部分
def run_experiment(adj_list, epsilon, eta):
    """
    运行LEDP最密子图算法
    :param adj_list: 图的邻接表
    :param epsilon: 隐私预算
    :param eta: 增长因子
    """
    print("运行LEDP最密子图算法...")
    subgraph, density = ledp_densest_subgraph(adj_list, epsilon, eta)
    print(f"最密子图节点集合: {subgraph}")
    print(f"估计密度: {density}")

# 主函数
if __name__ == "__main__":
    # 参数设置
    prefix = './datasets/Facebook/facebook/'
    files = ['414.edges', '107.edges']
    epsilon = 3.0  # 隐私预算
    eta = 3  # 增长因子

    print("读取图文件...")
    try:
        adj_list = read_facebook_graph(prefix, files[0])
    except Exception as e:
        print(f"读取图文件失败: {e}")
        exit(1)

    print("运行实验...")
    run_experiment(adj_list, epsilon, eta)
