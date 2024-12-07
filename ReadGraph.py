import networkx as nx
import os
import pandas as pd

def readFacebook(prefix,file):
    G = nx.Graph()
    file_path = os.path.join(prefix, file)  # 构建完整的文件路径
    # 读取每个文件并添加到图中
    G = nx.read_edgelist(file_path, nodetype=int, create_using=G)
    return G

def read_ca_astroph(file_path):
    # 初始化一个无向图
    G = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            # 跳过注释行
            if line.startswith("#"):
                continue
            # 从每一行中提取节点对
            from_node, to_node = map(int, line.strip().split())
            # 在图中添加边
            G.add_edge(from_node, to_node)
    return G

def read_ca_GrQC(file_path):
    # 初始化一个无向图
    G = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            # 跳过注释行
            if line.startswith("#"):
                continue
            # 从每一行中提取节点对
            from_node, to_node = map(int, line.strip().split())
            # 在图中添加边
            G.add_edge(from_node, to_node)
    return G

def read_twitch_graph(file_path):

    # 读取 CSV 文件
    edge_data = pd.read_csv(file_path)
    # 初始化无向图
    G = nx.Graph()
    # 将边添加到图中
    G.add_edges_from(zip(edge_data['from'], edge_data['to']))
    return G

def read_squirrel_graph(file_path):

    # 读取 CSV 文件
    edge_data = pd.read_csv(file_path)

    # 初始化无向图
    G = nx.Graph()

    # 将边添加到图中
    G.add_edges_from(zip(edge_data['id1'], edge_data['id2']))

    return G

