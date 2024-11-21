import numpy as np
# 引入上面的SEQDENSEDP和相关函数
def SEQDENSEDP(graph, epsilon, delta):
    n = len(graph)  # number of nodes
    epsilon_prime = epsilon / (4 * np.log(np.e / delta))
    S = list(graph.nodes)
    candidate_subgraphs = []  # 用于保存候选子集（子图对象）

    # Iteratively remove nodes with probability proportional to their degree using Exponential Mechanism
    for t in range(n):
        current_subgraph = graph.subgraph(S)
        degrees = {v: len(list(current_subgraph.neighbors(v))) for v in S}
        # Use the exponential mechanism to choose a node with probability proportional to exp(-epsilon_prime * degree)
        scores = np.array([-degrees[v] for v in S])  # Here we use negative degree as score (lower is better)
        probabilities = np.exp(epsilon_prime * scores)
        probabilities /= probabilities.sum()  # Normalize probabilities
        removed_node = np.random.choice(S, p=probabilities)
        S.remove(removed_node)

        # Add the current subgraph (as node list) to the candidate subgraph list
        candidate_subgraphs.append(S.copy())  # 存储节点列表

    # Calculate the density of each candidate subgraph
    subgraph_densities = [(i, calculate_density(graph.subgraph(subgraph))) for i, subgraph in
                          enumerate(candidate_subgraphs)]

    # Use the exponential mechanism to select the best subgraph based on their densities
    densities = np.array([density for _, density in subgraph_densities])
    probabilities = np.exp(epsilon * densities / 2)
    probabilities /= probabilities.sum()  # Normalize probabilities

    # Select the index of a subgraph from the candidate subgraphs with probability proportional to e^(epsilon * density / 2)
    selected_index = np.random.choice([i for i, _ in subgraph_densities], p=probabilities)
    best_subgraph = graph.subgraph(candidate_subgraphs[selected_index])
    density = len(list(best_subgraph.edges))/max(len((best_subgraph.nodes)),1)

    # Return the selected subgraph (as a graph object)
    return graph.subgraph(candidate_subgraphs[selected_index]),density


def calculate_density(subgraph):
    # density = (#edges in subgraph) / (#nodes in subgraph)
    num_nodes = len(subgraph.nodes)
    if num_nodes == 0:
        return 0
    return len(subgraph.edges) / num_nodes
# 运行测试