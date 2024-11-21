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
    #比较两个NetworkX子图的相似度
    # 将子图的节点集合转换为集合类型
    nodes_set1 = set(subgraph1.nodes())
    nodes_set2 = set(subgraph2.nodes())

    # 计算Jaccard相似度指数
    similarity = jaccard_similarity(nodes_set1, nodes_set2)
    return similarity
