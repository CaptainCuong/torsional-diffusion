import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch

def tree_converter_dfs(data):
    '''
    Args:
        data: (torch_geometric.data.data.Data)
            attrs: ['x', 'edge_index', 'edge_attr', 'z', 'canonical_smi', 'mol', 'pos', 'weights']
    Return:
        Directed edges of a tree
    '''
    networkx = to_networkx(data)
    tree = nx.DiGraph()
    [tree.add_node(node) for node in networkx.nodes()]
    traversed = []
    traversing = [(0,0)]
    converted = False
    while traversing:
        # Process a waiting edge
        u, v = traversing.pop(-1)
        if v in traversed:
            converted = True
            continue
        tree.add_edge(u,v)
        traversed.append(v)
        nbrs = list(networkx.neighbors(v))
        nbrs.reverse()
        # Add new edges
        for nbr in nbrs:
            if nbr not in traversed:
                traversing.append((v,nbr))

    tree.remove_edge(0,0)
    assert len(traversed) == len(tree.nodes()), str(traversed) + ' != ' + str(tree.nodes())
    assert len(tree.nodes)-1 == len(tree.edges())
    assert networkx.number_of_nodes() == tree.number_of_nodes(), data.canonical_smi
    return tree, converted

def tree_converter_bfs(data):
    networkx = to_networkx(data)
    tree = nx.DiGraph()
    [tree.add_node(node) for node in networkx.nodes()]
    traversed = []
    traversing = [0]
    converted = False
    while traversing:
        # Process a waiting edge
        v = traversing.pop(0)
        traversed.append(v)
        nbrs = list(networkx.neighbors(v))
        
        # Add new traversing node & add edges for the tree
        for nbr in nbrs:
            if nbr not in traversed and nbr not in traversing:
                '''
                not in traversed: check if nbr is a parent node
                not in traversing: check if nbr is not in the waiting list
                '''
                traversing.append(nbr)
                tree.add_edge(v,nbr)
            else:
                converted = True
    
    assert len(traversed) == len(tree.nodes()), str(traversed) + ' != ' + str(tree.nodes())
    assert len(tree.nodes)-1 == len(tree.edges())
    assert networkx.number_of_nodes() == tree.number_of_nodes(), data.canonical_smi
    return tree, converted

def break_loop(data, tree):
    '''
    Data(x=[15, 44], edge_index=[2, 30], edge_attr=[30, 4], z=[15], canonical_smi='C#CC#CC#CC1CC1', 
    mol=<rdkit.Chem.rdchem.Mol object at 0x000002742AD30D10>, pos=[1], weights=[1])
    '''
    tree_edge = [[u,v] for u, v in tree.edges]
    G_edge = data.edge_index.T.tolist()
    tree_mask = []
    converted = False
    for i in range(0, len(G_edge), 2):
        if G_edge[i] in tree_edge or G_edge[i+1] in tree_edge:
            tree_mask.append(True)
            tree_mask.append(True)
        else:
            tree_mask.append(False)
            tree_mask.append(False)
            converted = True
    
    data.edge_index = data.edge_index.T[tree_mask].T
    data.edge_attr = data.edge_attr[tree_mask]