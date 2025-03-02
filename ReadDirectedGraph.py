import gzip
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
def ReadBitcoin(file_path):
    """
    读取Bitcoin OTC数据集并构建有向图。

    参数:
    file_path (str): 数据集文件路径，可以是压缩文件（.gz）或解压后的文件（.csv）。

    返回:
    G (networkx.DiGraph): 构建的有向图。
    """
    # 创建一个空的有向图
    G = nx.DiGraph()

    # 使用 Pandas 一次性读取边列表
    if file_path.endswith(".gz"):
        df = pd.read_csv(file_path, sep=",", comment="#", header=None, names=["source", "target", "timestamp"])
    else:
        df = pd.read_csv(file_path, sep=",", comment="#", header=None, names=["source", "target", "timestamp"])

    # 将数据框转换为有向图
    G = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.DiGraph())

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

# 文件路径
file_path = 'DirectedDatasets/soc-sign-bitcoinotc.csv.gz'

# 使用 gzip 打开文件
with gzip.open(file_path, 'rt') as file:
    # 读取 CSV 文件内容
    data = pd.read_csv(file, header=None)

# 为数据框添加列名
data.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']

# 创建一个有向图
G = nx.DiGraph()

# 添加边和权重
for _, row in data.iterrows():
    G.add_edge(row['SOURCE'], row['TARGET'], weight=row['RATING'])

# 随机选取 100 个节点
nodes = list(G.nodes())
random_nodes = random.sample(nodes, min(100, len(nodes)))

# 创建一个子图，只包含随机选取的节点及其边
subgraph_nodes = set(random_nodes)
subgraph_edges = [(u, v) for u, v in G.edges() if u in subgraph_nodes and v in subgraph_nodes]
subgraph = G.edge_subgraph(subgraph_edges).copy()

# 绘制子图
pos = nx.spring_layout(subgraph)  # 使用弹簧布局
plt.figure(figsize=(12, 12))
nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
plt.title('Subgraph of 100 Random Nodes from Bitcoin OTC Network')
plt.show()