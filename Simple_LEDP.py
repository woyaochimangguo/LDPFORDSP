#baseline2 based on the paper "Almost Tight Bounds for Differentially Private Densest Subgraph"
import numpy as np
def geometric_noise(epsilon):
    """
    Generate symmetric geometric noise with parameter epsilon.
    """
    p = np.exp(-epsilon)
    u = np.random.uniform(0, 1)
    noise = np.sign(u - 0.5) * np.floor(np.log(1 - 2 * abs(u - 0.5)) / np.log(p))
    return int(noise)

def node_calculate_noisy_degree(graph, node, current_nodes, epsilon):
    """
    Each node independently calculates its noisy degree, including self-loops.
    """
    true_degree = sum(1 for neighbor in graph.neighbors(node) if neighbor in current_nodes)
    # Include self-loop explicitly if the node has one
    if graph.has_edge(node, node):
        true_degree += 1
    noisy_degree = true_degree + geometric_noise(epsilon)
    return noisy_degree

def simple_epsilon_ledp(graph, total_epsilon, eta):
    """
    Simple ε-LEDP algorithm for dense subgraph discovery.

    Parameters:
        graph (networkx.Graph): Input graph in networkx format.
        total_epsilon (float): Total privacy budget.
        eta (float): Noise threshold scaling parameter.

    Returns:
        list: Subset of nodes with the largest estimated density.
    """
    # Step 1: Precompute total iterations and per-iteration epsilon
    n = graph.number_of_nodes()  # Number of nodes in the graph
    k = int(np.ceil((1 / eta) * np.log(n)))  # Total iterations
    epsilon_per_iteration = total_epsilon / (2 * k)  # Privacy budget per iteration

    # Initial setup
    S = list(graph.nodes())
    subsets = []  # Store subsets
    densities = []  # Store estimated densities

    for iteration in range(1, k + 1):
        if not S:
            break

        # Step 3: Each node independently calculates its noisy degree
        noisy_degrees = {}
        for v in S:
            noisy_degrees[v] = node_calculate_noisy_degree(graph, v, S, epsilon_per_iteration)

        # Step 5: Compute density estimate at the central server
        estimated_density = sum(noisy_degrees.values()) / (2 * len(S))
        subsets.append(S.copy())
        densities.append(estimated_density)

        # Step 6: Compute noise threshold at the central server
        avg_noisy_degree = sum(noisy_degrees.values()) / len(S)
        threshold = (1 + eta) * avg_noisy_degree

        # Step 7: Central server identifies low-degree nodes
        low_degree_nodes = [v for v, d in noisy_degrees.items() if d <= threshold]

        # Step 8: Update S and remove low-degree nodes
        S = [v for v in S if v not in low_degree_nodes]

        # Update the graph to remove edges associated with deleted nodes
        graph = graph.subgraph(S).copy()

    # Step 10: Select subset with the largest density
    max_density_index = np.argmax(densities)
    return subsets[max_density_index]
