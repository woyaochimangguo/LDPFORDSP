import networkx as nx
import numpy as np
import random
from collections import defaultdict

def ledp_densest_subgraph(G, epsilon, psi):
    """
    Implementation of ε-LEDP Densest Subgraph Algorithm.

    Parameters:
        G (networkx.Graph): Input graph.
        epsilon (float): Privacy parameter (0 < ε < 1).
        psi (float): Approximation parameter (0 < ψ < 1).

    Returns:
        S (set): Approximate densest subgraph node set.
    """
    n = len(G.nodes())
    log_n = int(np.ceil(np.log2(n)))
    adjacency_list = {node: list(G.neighbors(node)) for node in G.nodes()}

    # Initialize L data structure
    L = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for r in range(2 * log_n):
        for g in range(2 * log_n):
            for i in range(n):
                L[g][r][i] = 0

    for r in range(2 * log_n - 1):
        for i, node in enumerate(G.nodes()):
            for g in range(2 * log_n):
                L[g][r + 1][i] = L[g][r][i]
                if L[g][r][i] == r:
                    # Compute the number of neighbors at level r in group g
                    U_ig = sum(
                        1 for neighbor in adjacency_list[node]
                        if L[g][r][list(G.nodes()).index(neighbor)] == r
                    )
                    # Add geometric noise
                    noise = np.random.geometric(1 - np.exp(-epsilon / (8 * log_n ** 2))) - 1
                    U_ig_noisy = U_ig + noise

                    # Check if node should move to the next level
                    if U_ig_noisy > (1 + psi) * g:
                        L[g][r + 1][i] = L[g][r][i] + 1

    # Peeling step to find densest subgraph
    max_density = 0
    best_subgraph = set()
    for g in range(2 * log_n):
        for r in range(2 * log_n - 1, -1, -1):
            subgraph_nodes = {
                node for i, node in enumerate(G.nodes())
                if L[g][r][i] > 0
            }
            if not subgraph_nodes:
                continue

            noisy_degree_sum = sum(
                len(list(G.neighbors(node))) + np.random.geometric(1 - np.exp(-epsilon / (8 * log_n ** 2))) - 1
                for node in subgraph_nodes
            )
            density = noisy_degree_sum / (2 * len(subgraph_nodes))
            if density > max_density:
                max_density = density
                best_subgraph = subgraph_nodes

    return best_subgraph