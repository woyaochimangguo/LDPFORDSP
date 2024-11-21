from Our_algorithm import *
from datetime import datetime
from baseline_1_GreedyPeeling_withoutDP import charikar_peeling
from ReadGraph import *
from baseline_2_GreedyPeeling_UnderDP import SEQDENSEDP
from measurements import *
prefix = './datasets/Facebook/facebook/'
files = ['414.edges', '107.edges']
epsilon = 20.0
delta = 1e-9
# 读取文件并添加边
G = readFacebook(prefix, files[0])

# 原始图的稠密子图提取
print("Original graph - Greedy peeling method")
start = datetime.now()
dense_subgraph, density = charikar_peeling(G)
print('Run time:', datetime.now() - start)
print("Nodes in dense subgraph:", dense_subgraph.nodes())
print("Density of the dense subgraph in original graph:", density)

#中心化差分隐私的最密子图提取
print("Original graph - DP Greedy peeling method")
start = datetime.now()
dense_subgraph_DP, density_DP = SEQDENSEDP(G,epsilon,delta)
print('Run time:', datetime.now() - start)
print("Nodes in dense subgraph:", dense_subgraph_DP.nodes())
print("Density of the dense subgraph in original graph with DP:", density_DP)
jaccard_similarity =  compare_subgraph_similarity(dense_subgraph,dense_subgraph_DP)
print("Jaccard index of the two dense subgraphs:",jaccard_similarity)

#我们的算法差分隐私后的噪声图的稠密子图提取
print("\nNoisy graph - Greedy peeling method")
start = datetime.now()
dense_subgraph_ldpdsp, density_ldpdsp = ldp_charikar_peeling(G, epsilon)
jaccard_similarity_LDP =  compare_subgraph_similarity(dense_subgraph,dense_subgraph_ldpdsp)
print('Run time:', datetime.now() - start)
print("Nodes in dense subgraph:", dense_subgraph_ldpdsp.nodes())
print("Density of the dense subgraph in LDP:", density_ldpdsp)
print("Jaccard index of the two dense subgraphs:",jaccard_similarity_LDP)

print("epsilon:",epsilon)
print("baseline1",density)
print("baseline2",density_DP)
print("ours",density_ldpdsp)