def densest_subgraph_LDP_real_1(G, epsilon):
    """LDP-Densest-Subgraph-Directed 算法，添加 lock 机制"""
    n = len(G.nodes())
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)

    # 生成扰动图
    perturbed_out_neighbors = apply_rr_out_neighbors(G, epsilon)
    G_prime = build_perturbed_graph(perturbed_out_neighbors)

    S, T = set(G_prime.nodes()), set(G_prime.nodes())
    rho_max = 0
    S_best, T_best = S.copy(), T.copy()

    # 计算初始的 dout_e 和 din_e
    dout_n = {v: G_prime.out_degree(v) for v in G_prime.nodes()}
    din_n = {v: G_prime.in_degree(v) for v in G_prime.nodes()}
    dout_e = {v: (dout_n[v] - n * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}
    din_e = {v: (din_n[v] - n * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}

    # 维护锁定的节点集合
    locked_out = set()  # 记录出度已锁定的节点
    locked_in = set()   # 记录入度已锁定的节点

    while S and T:
        # 计算阈值
        threshold = (n * (1 - p)) / (2 * p - 1)
        error_tolerance = 1e-6

        # 找出 dout_e 和 din_e 最小的两个节点（不能已锁定 & 不能接近 threshold）
        min_out_node = min(
            (v for v in S | T if v not in locked_out and not math.isclose(dout_e[v], threshold, abs_tol=error_tolerance)),
            key=lambda v: dout_e[v],
            default=None
        )
        print("min_out_node",min_out_node)
        min_in_node = min(
            (v for v in S | T if v not in locked_in and not math.isclose(din_e[v], threshold, abs_tol=error_tolerance)),
            key=lambda v: din_e[v],
            default=None
        )
        print("min_in_node",min_in_node)

        # 如果没有可选节点，说明所有未锁定的节点的度数都接近 threshold，退出循环
        if min_out_node is None and min_in_node is None:
            break

        # 选择出度或入度最小的节点
        if min_out_node is not None and (min_in_node is None or dout_e[min_out_node] < din_e[min_in_node]):
            u = min_out_node
            G_prime.remove_edge(u, v)
            # 删除出边
            for v in list(G_prime.successors(u)):
                if v not in locked_in:  # 只有未锁定的节点才更新
                    if (u, v) in G_prime.edges:
                        din_e[v] -= p / (2 * p - 1)
                    else:
                        din_e[v] -= (p - 1) / (2 * p - 1)
            # 锁定出度
            print("锁定出度最小节点",u)
            dout_e[u] = threshold
            locked_out.add(u)

        else:
            u = min_in_node
            G_prime.remove_edge(v, u)
            # 删除入边
            for v in list(G_prime.predecessors(u)):
                if v not in locked_out:  # 只有未锁定的节点才更新
                    if (v, u) in G_prime.edges:
                        dout_e[v] -= p / (2 * p - 1)
                    else:
                        dout_e[v] -= (p - 1) / (2 * p - 1)
            # 锁定入度
            print("锁定入度最小节点",u)
            din_e[u] = threshold
            locked_in.add(u)

        # 判断 u 是否满足删除条件（允许误差范围）
        if math.isclose(dout_e[u], threshold, abs_tol=error_tolerance) and math.isclose(din_e[u], threshold, abs_tol=error_tolerance):
            G_prime.remove_node(u)
            print("删除节点",u)

        # 更新 S 和 T
        S = {v for v in G_prime.nodes() if dout_e[v] > 0}
        T = {v for v in G_prime.nodes() if din_e[v] > 0}

        # 计算密度
        E_ST = sum(1 for u in S for v in T if G_prime.has_edge(u, v))
        rho = (E_ST - (1 - p) * len(S) * len(T)) / ((2 * p - 1) * (len(S) * len(T)) ** 0.5)

        if rho > rho_max:
            rho_max = rho
            S_best, T_best = S.copy(), T.copy()

    return S_best, T_best


def densest_subgraph_LDP_nolock(G, epsilon):
    """LDP-Densest-Subgraph-Directed 算法"""
    n = len(G.nodes())
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    perturbed_out_neighbors = apply_rr_out_neighbors(G, epsilon)
    G_prime = build_perturbed_graph(perturbed_out_neighbors)

    S, T = set(G_prime.nodes()), set(G_prime.nodes())
    rho_max = 0
    S_best, T_best = S, T

    # 计算初始的 dout_e 和 din_e
    dout_n = {v: G_prime.out_degree(v) for v in G_prime.nodes()}
    din_n = {v: G_prime.in_degree(v) for v in G_prime.nodes()}
    dout_e = {v: (dout_n[v] - len(G.nodes()) * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}
    din_e = {v: (din_n[v] - len(G.nodes()) * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}

    while S and T:
        # 计算阈值
        threshold = (n * (1 - p)) / (2 * p - 1)
        error_tolerance = 1e-6
        min_out_node = min(
            (v for v in S | T if not math.isclose(dout_e[v], threshold, abs_tol=error_tolerance)),
            key=lambda v: dout_e[v],
            default=None
        )
        print("min_out_node",min_out_node,dout_e[min_out_node])
        min_in_node = min(
            (v for v in S | T if not math.isclose(din_e[v], threshold, abs_tol=error_tolerance)),
            key=lambda v: din_e[v],
            default=None
        )
        print("min_in_node",min_in_node,din_e[min_in_node])
        # 如果两者都为空，说明所有节点的度数都等于 threshold，退出循环
        if min_out_node is None and min_in_node is None:
            break

        # 选择出度或入度最小的节点
        if min_out_node is not None and (min_in_node is None or dout_e[min_out_node] < din_e[min_in_node]):
            print("出度最小")
            u = min_out_node
            dout_e[u] = threshold
            # 删除出边
            for v in list(G_prime.successors(u)):
                if (u, v) in G_prime.edges:
                    din_e[v] -= p / (2 * p - 1)
                else:
                    din_e[v] -= (p - 1) / (2 * p - 1)

        else:
            u = min_in_node
            print("入度最小")
            din_e[u] = threshold
            # 删除入边
            for v in list(G_prime.predecessors(u)):
                if (v, u) in G_prime.edges:
                    dout_e[v] -= p / (2 * p - 1)
                else:
                    dout_e[v] -= (p - 1) / (2 * p - 1)

        # 判断 u 是否满足删除条件
        if math.isclose(dout_e[u], threshold, abs_tol=error_tolerance) and math.isclose(din_e[u], threshold, abs_tol=error_tolerance):
            G_prime.remove_node(u)

        # 更新 S 和 T
        S = {v for v in G_prime.nodes() if dout_e[v] > 0}
        T = {v for v in G_prime.nodes() if din_e[v] > 0}

        # 计算密度
        E_ST = sum(1 for u in S for v in T if G_prime.has_edge(u, v))
        rho = (E_ST - (1 - p) * len(S) * len(T)) / ((2 * p - 1) * (len(S) * len(T)) ** 0.5)

        if rho > rho_max:
            rho_max = rho
            S_best, T_best = S.copy(), T.copy()

    return S_best, T_best


def densest_subgraph_LDP1(G, epsilon):
    """LDP-Densest-Subgraph-Directed 算法，使用 remove_edges_from 进行边删除"""
    n = len(G.nodes())
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    perturbed_out_neighbors = apply_rr_out_neighbors(G, epsilon)
    G_prime = build_perturbed_graph(perturbed_out_neighbors)

    S, T = set(G_prime.nodes()), set(G_prime.nodes())
    rho_max = 0
    S_best, T_best = S.copy(), T.copy()

    # 计算初始 dout_e 和 din_e
    dout_n = {v: G_prime.out_degree(v) for v in G_prime.nodes()}
    din_n = {v: G_prime.in_degree(v) for v in G_prime.nodes()}
    dout_e = {v: (dout_n[v] - n * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}
    din_e = {v: (din_n[v] - n * (1 - p)) / (2 * p - 1) for v in G_prime.nodes()}

    locked_nodes = set()  # 记录已锁定的节点

    while S and T:
        threshold = (n * (1 - p)) / (2 * p - 1)
        error_tolerance = 1e-6

        # 选择度最小的节点（排除已锁定节点）
        min_out_node = min(
            (v for v in S | T if v not in locked_nodes and not math.isclose(dout_e[v], threshold, abs_tol=error_tolerance)),
            key=lambda v: dout_e[v],
            default=None
        )
        min_in_node = min(
            (v for v in S | T if v not in locked_nodes and not math.isclose(din_e[v], threshold, abs_tol=error_tolerance)),
            key=lambda v: din_e[v],
            default=None
        )

        if min_out_node is None and min_in_node is None:
            break  # 没有可删除的节点，退出循环

        if min_out_node is not None and (min_in_node is None or dout_e[min_out_node] < din_e[min_in_node]):
            u = min_out_node
            print(f"删除出度最小的节点 {u}")
            edges_to_remove = list(G_prime.out_edges(u))
            # **更新所有受影响节点的入度估计**
            for _, v in edges_to_remove:
                if (u, v) in G_prime.edges:  # 真实边
                    din_e[v] -= p / (2 * p - 1)
                else:  # 噪声边
                    din_e[v] -= (p - 1) / (2 * p - 1)
            G_prime.remove_edges_from(edges_to_remove)
            dout_e[u] = threshold  # 锁定度数
            locked_nodes.add(u)

        else:
            u = min_in_node
            print(f"删除入度最小的节点 {u}")
            edges_to_remove = list(G_prime.in_edges(u))
            # **用 remove_edges_from 一次性删除所有入边**
            # **更新所有受影响节点的出度估计**
            for v, _ in edges_to_remove:
                if (v, u) in G_prime.edges:  # 真实边
                    dout_e[v] -= p / (2 * p - 1)
                else:  # 噪声边
                    dout_e[v] -= (p - 1) / (2 * p - 1)
            G_prime.remove_edges_from(edges_to_remove)
            din_e[u] = threshold  # 锁定度数
            locked_nodes.add(u)

        # 如果 u 的出度和入度都接近 threshold，则删除 u
        if math.isclose(dout_e[u], threshold, abs_tol=error_tolerance) and math.isclose(din_e[u], threshold, abs_tol=error_tolerance):
            print(f"删除节点 {u}（所有度数达到阈值）")
            G_prime.remove_node(u)

        # 更新 S 和 T
        S = {v for v in G_prime.nodes() if dout_e[v] > threshold}
        T = {v for v in G_prime.nodes() if din_e[v] > threshold}

        # 计算密度
        E_ST = sum(1 for u in S for v in T if G_prime.has_edge(u, v))
        rho = (E_ST - (1 - p) * len(S) * len(T)) / ((2 * p - 1) * (len(S) * len(T)) ** 0.5)

        if rho > rho_max:
            rho_max = rho
            S_best, T_best = S.copy(), T.copy()

    return S_best, T_best
