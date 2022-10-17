import networkx as nx

'''
Rotate smaller component according to larger component

Preprocessing:
	1) Delete edges in ring with out-deg == 2
'''

for u, v in Bond.values():
	G.remove_edge(bond)
	if len(nx.connected_components(G)) == 1:
		all_simple_path(G, u, v)