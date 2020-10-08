import networkx as nx

# 创建有向图
G = nx.DiGraph()
# 有向图之间边的关系
edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]
for edge in edges:
    G.add_edge(edge[0], edge[1])
# 这里的alpha是阻尼因子，这里是1，表示我们都是用跳转链接，不直接输入网址的那种。
pagerank_list = nx.pagerank(G, alpha=1)
print("pagerank值是：\n", pagerank_list)
