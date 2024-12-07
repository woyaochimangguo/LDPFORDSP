import numpy as np
import networkx as nx

def laplace_mechanism(value, epsilon, sensitivity=1):
    """添加拉普拉斯噪声"""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

def algorithm_7_k_core_decomposition(G, epsilon, eta=1):
    """
    改进的本地差分隐私 k-核分解算法
    输入:
        G: 图 (NetworkX 格式)
        epsilon: 隐私参数
        eta: 用于调整 ψ 和 λ 的动态参数
    输出:
        核数估计值 ˆk(v) 的字典
    """
    n = G.number_of_nodes()
    log_rounds = int(4 * (np.log2(n) ** 2))  # 动态计算轮次数
    nodes = list(G.nodes())
    levels = {v: 0 for v in nodes}  # 每个节点的初始 level
    results = {}  # 最终核数结果

    # 动态计算 ψ 和 λ
    psi = 0.1 * eta
    lambda_ = 2 * (30 - eta) * eta / ((eta + 10) ** 2)

    # 每个节点的阈值噪声
    thresholds = {v: laplace_mechanism(0, 4 / epsilon) for v in nodes}

    # 每轮的节点层级记录
    L = {v: 0 for v in nodes}

    for r in range(log_rounds):
        updated_nodes = set()
        for v in nodes:
            if L[v] == r:  # 检查节点是否在当前 level
                # 计算当前邻居的有效数量
                neighbors_at_r = [u for u in G.neighbors(v) if L[u] == r]
                noisy_degree = laplace_mechanism(len(neighbors_at_r), 8 / epsilon, sensitivity=2)

                # 检查是否满足移除条件
                threshold = (1 + psi)**(r / (2 * np.log(n))) + thresholds[v]
                if noisy_degree <= threshold:
                    L[v] += 1
                updated_nodes.add(v)

        # 更新所有被移除的节点
        for v in updated_nodes:
            L[v] += 1

    # Step 21: Curator publishes Lr+1
    # Step 22-27: Compute final core numbers
    xx = 0
    for v in nodes:
        xx = xx+1
        # Step 23: Find the highest level ℓ′ where Lℓ′ [i] = ℓ′ or 0
        ℓ_prime = L[v] if L[v] >= 0 else 0

        # Step 24: Calculate approximate core number using the formula
        core_number = (2 + lambda_) * (1 + psi)**max(
            (ℓ_prime + 1) / (4 * np.ceil(np.log(1+psi**xx)))-1, 1
        )

        # Step 25: Add the computed core number to the result set
        results[v] = core_number

    # Step 27: Return the set of core numbers
    return results

def algorithm_8_densest_subgraph(G, epsilon, c_prime=100000, eta=2):
    """
    基于 k-核分解计算最密子图
    输入:
        G: 图 (NetworkX 格式)
        epsilon: 隐私参数
        c_prime: 常量参数
        eta: 用于调整 ψ 和 λ 的动态参数
    输出:
        最密子图的节点集合 S
    """
    # 使用算法 7 计算核数
    core_numbers = algorithm_7_k_core_decomposition(G, epsilon, eta)

    # 找到最大核数
    k_max = max(core_numbers.values())

    # 筛选满足条件的节点
    threshold = k_max - c_prime * (np.log(G.number_of_nodes()) / epsilon)
    densest_subgraph = [v for v, k in core_numbers.items() if k >= threshold]

    return densest_subgraph

# 测试代码