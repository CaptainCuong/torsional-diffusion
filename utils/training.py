import numpy as np
from tqdm import tqdm
import torch
import diffusion.torus as torus


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):
        '''
        DataBatch(x=[458, 44], edge_index=[2, 866], edge_attr=[866, 4], z=[458], canonical_smi=[32], mol=[32], pos=[458, 3], weights=[32], edge_mask=[866], mask_rotate=[32], node_sigma=[458], edge_rotate=[197], batch=[458], ptr=[33])
            x: [no_nodes, no_features] # one hot encoding
            edge_index: [2, no_edges] # Directional. Ex: [[1, 0], [0, 1]]
            edge_attr: [no_edges, 4]
            z: [no_nodes]
            canonical_smi: [no_mols]
            mol: [no_mols]
            pos: [no_nodes, 3]
            weights: [no_mols]
            edge_mask=[no_edges]
            mask_rotate=[no_mols] # list
            node_sigma=[no_nodes]
            edge_rotate=[197] 
            batch=[no_nodes] # A batch contains several(32) graphs.
                            The attribute batch separates atoms in the same graph.
            ptr=[num_graphs+1] # Batch/Graph separator

            !!! 
            no_nodes: number of atoms in an entire batch
            edge_mask:
                Rotatable bonds
                True if 1) The graph is separated after eliminating edge (not ring) AND
                           Smaller component has at least 2 nodes (not terminal bond)
                        2) Edge with to_node in the larger component

                    xxx -----> xxxxxxxx (True) (rotatable_edge)
                    xxx <----- xxxxxxxx (False)

            mask_rotate[i]:
                Rotated nodes
                Shape:
                    Size(No_rotatable_bonds, No_nodes)
                Indexing:
                    mask_rotate[i][edge_idx][node_idx]
                True if:
                    A node in the smaller component separated by a rotatable bond
                
                xxx -----> xxxxxxxx
                TTT -----> FFFFFFFF
                Constraint:
                    mask_rotate[idx_rotatable_edge, u]: False
                    mask_rotate[idx_rotatable_edge, v]: True
                    u, v = rotatable_edge[0], rotatable_edge[1]
            node_sigma:
                Nodes in the same molecule have the same value of sigma
                Random from node_sigma in TorsionNoiseTransform
        '''
        data = data.to(device)
        optimizer.zero_grad()

        data = model(data)
        '''
        edge_pred=[No_rotatable_bonds], 
        edge_sigma=[No_rotatable_bonds]
        '''
        pred = data.edge_pred
        try:
            score = torus.score(
                data.edge_rotate.cpu().numpy(),
                data.edge_sigma.cpu().numpy())
        except:
            raise
        score = torch.tensor(score, device=pred.device)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm, device=pred.device)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg


@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):

        data = data.to(device)
        data = model(data)
        pred = data.edge_pred.cpu()

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg

