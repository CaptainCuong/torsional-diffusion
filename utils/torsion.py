import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def get_transformation_mask(pyg_data):
    '''
    Input: torch_geometric graph
    Output: mask_edges, mask_rotate
    '''
    # Converts a torch_geometric.data.Data instance to a networkx.Graph 
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2): # Increase by 2, so do not analyze an edge twice
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected() # Convert to undirected graph
        G2.remove_edge(*edges[i]) # Remove the edges[i]
        if not nx.is_connected(G2): # If not connected (edges[i] is not in a ring) (bridge)
            l = list(sorted(nx.connected_components(G2), key=len)[0]) # Nodes in smallest connected component
            if len(l) > 1: # Has at least 2 nodes
                if edges[i, 0] in l:
                    to_rotate.append([]) # Add False(for edges[i]), True(for edges[i+1])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l) # Add True(for edges[i]), False(for edges[i+1])
                    to_rotate.append([])
                continue
                '''
                Edge with to_node in the larger component is True
                Edge with to_node in the smaller component is False
                '''
        to_rotate.append([]) # If connected (ring) or the smallest connected component has 1 node (terminal bond), add 2 empty lists
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    '''
    mask_edges: [len(G.edges())]
    True if 1) Smaller component has at least 2 nodes AND
               The target node in the smaller component
            2) Edge with to_node in the larger component
        
        xxx -----> xxxxxxxx (True)
        xxx <----- xxxxxxxx (False)
    '''
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool) # [No_rotatable_bonds][No_nodes]
    idx = 0 # index of a rotatable bond
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True # to_rotate[i]: nodes in the smaller component
            # Nodes in the smaller component separated by a rotatable bond is True
            idx += 1
    '''
    mask_rotate:
        Shape:
            Size(No_rotatable_bonds, No_nodes)
        True if:
            A node in the smaller component separated by a rotatable bond
        
        xxx -----> xxxxxxxx
        TTT -----> FFFFFFFF
    '''

    return mask_edges, mask_rotate


def get_distance_matrix(pyg_data, mask_edges, mask_rotate):
    G = to_networkx(pyg_data, to_undirected=False)
    N = G.number_of_nodes()
    edge_distances = []
    for i, e in enumerate(pyg_data.edge_index.T.numpy()[mask_edges]):
        v = e[1]
        d = nx.shortest_path_length(G, source=v)
        d = np.asarray([d[j] for j in range(N)])
        d = d - 1 + mask_rotate[i]
        edge_distances.append(d)

    edge_distances = np.asarray(edge_distances)
    return edge_distances


def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    '''
    Update positions of a conformer
    Parameters:
        pos: [no_nodes]
            position of molecules in a conformer
        edge_index: [No_rotatable_bonds, 2]
            list of rotatable edges
        torsion_updates: [No_rotatable_nodes]
            list of rotation angle
            -np.pi < angle < np.pi
    '''
    if type(pos) != np.ndarray: pos = pos.cpu().numpy() # Convert data type
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        # Create rotation matrix from rotation vector and a torsion update angle
        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    '''
    Parameters:
        data: 
            type: BatchData
    '''
    if type(data) is Data:
        return modify_conformer(data.pos, 
            data.edge_index.T[data.edge_mask], 
            data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]] # mask_rotate.shape[1]: number of nodes in that molecule
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer(pos, edges, mask_rotate, torsion_update) # Update new position
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new