import gzip
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
def ReadBitcoin(file_path):
    """加载 Bitcoin OTC 数据集并构建有向图"""
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f, header=None, usecols=[0, 1], names=['source', 'target'])
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, tgt = int(row['source']), int(row['target'])
        G.add_edge(src, tgt)
    return G

def ReadSocEpinions(file_path):
    """
    读取SOC-Epinions1数据集并构建有向图。

    参数:
    file_path (str): 数据集文件路径，可以是压缩文件（.gz）或解压后的文件（.txt）。

    返回:
    G (networkx.DiGraph): 构建的有向图。
    """
    # 创建一个空的有向图
    G = nx.DiGraph()

    # 使用 Pandas 一次性读取边列表
    if file_path.endswith(".gz"):
        df = pd.read_csv(file_path, sep="\s+", comment="#", header=None, names=["source", "target"])
    else:
        df = pd.read_csv(file_path, sep="\s+", comment="#", header=None, names=["source", "target"])

    # 将数据框转换为有向图
    G = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.DiGraph())

    return G
