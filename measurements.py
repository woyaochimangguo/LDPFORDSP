import networkx as nx
def jaccard_similarity(set1, set2):
    #计算两个集合的Jaccard相似度指数
    # 计算两个集合的交集
    intersection = len(set1.intersection(set2))
    # 计算两个集合的并集
    union = len(set1.union(set2))
    # 计算Jaccard相似度指数
    if union == 0:
        return 0  # 避免除以0的情况
    return intersection / union

def compare_subgraph_similarity(subgraph1, subgraph2):
    """
    计算两个子图的 Jaccard 相似度
    :param subgraph1: 第一个子图（集合或 NetworkX 图）
    :param subgraph2: 第二个子图（集合或 NetworkX 图）
    :return: Jaccard 相似度
    """
    # 确保处理的是集合
    if isinstance(subgraph1, nx.Graph):
        nodes_set1 = set(subgraph1.nodes())
    else:
        nodes_set1 = set(subgraph1)

    if isinstance(subgraph2, nx.Graph):
        nodes_set2 = set(subgraph2.nodes())
    else:
        nodes_set2 = set(subgraph2)

    # 计算 Jaccard 相似度
    intersection = nodes_set1 & nodes_set2
    union = nodes_set1 | nodes_set2
    if not union:  # 避免除以零
        return 0.0
    return len(intersection) / len(union)
