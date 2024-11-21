import networkx as nx
import os
#way 1 from IMDB dataset
def readIMDB():
    # 读取文件并添加边
    G = nx.Graph()
    with open("./datasets/IMDB/edges.csv", "r") as f:
        for line in f:
            # 去掉行尾换行符并按逗号分割每一行
            node1, node2 = map(int, line.strip().split(","))
            G.add_edge(node1, node2)
            return G

#way 2 from Facebook dataset
def readFacebook(prefix,file):
    G = nx.Graph()
    file_path = os.path.join(prefix, file)  # 构建完整的文件路径
    # 读取每个文件并添加到图中
    G = nx.read_edgelist(file_path, nodetype=int, create_using=G)
    return G
