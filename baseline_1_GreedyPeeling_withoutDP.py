def charikar_peeling(G):
    """
    Charikar's peeling algorithm to find a dense subgraph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    tuple: A tuple containing the dense subgraph and its density.
    """
    # Copy the original graph to avoid modifying it
    G = G.copy()

    # Initialize variables to track the best density and corresponding subgraph
    best_density = 0
    best_subgraph = None

    # Compute initial density
    def compute_density(G):
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        return num_edges / num_nodes if num_nodes > 0 else 0


    # Iteratively remove the node with the smallest degree
    while G.number_of_nodes() > 0:
        current_density = compute_density(G)
        if current_density > best_density:
            best_density = current_density
            best_subgraph = G.copy()
            print("current density is", current_density)
            print("current subgraph nodes is ",best_subgraph)
        # Find the node with the smallest degree
        min_degree_node = min(G.nodes, key=G.degree)
        G.remove_node(min_degree_node)

    return best_subgraph, best_density
