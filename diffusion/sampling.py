import random
from utils.featurization import featurize_mol, featurize_mol_from_smiles
from utils.torsion import *
from diffusion.likelihood import *
import torch, copy
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from utils.visualise import PDBFile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
still_frames = 10


def try_mmff(mol):
    '''
    Test whether it is able to apply the function, MMFFOptimizeMoleculeConfs.
    '''
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        '''
        RETURNS: a list of (not_converged, energy) 2-tuples.
            If not_converged is 0 the optimization converged for that conformer.
        '''
        return True
    except Exception as e:
        return False


def get_seed(smi, seed_confs=None, dataset='drugs'):
    '''
    Convert smile to graph
    Get edge_mask, mask_rotate
    '''
    if seed_confs: ##### Be not used
        if smi not in seed_confs:
            print("smile not in seeds", smi)
            return None, None
        mol = seed_confs[smi][0]
        data = featurize_mol(mol, dataset)

    else:
        mol, data = featurize_mol_from_smiles(smi, dataset=dataset) # Return graph from smile
        if not mol:
            return None, None
    data.edge_mask, data.mask_rotate = get_transformation_mask(data) # Get rotatable bond
    data.edge_mask = torch.tensor(data.edge_mask)
    return mol, data


def embed_seeds(mol, data, n_confs, single_conf=False, smi=None, embed_func=None, seed_confs=None, pdb=None, mmff=False):
    '''
    Separate conformers (1 mol, 1 conformer)
    Return
        conformers: List of data.Data objects
        Each data.Data object contains only a conformer
        The conformers are initialized with the positions of the first conformer in $data

    Example: 
    Input: mol1, (conf1, conf2, conf3)
    Output: [(mol1, conf1), (mol1, conf2), (mol1, conf3)]
    '''
    if not seed_confs: # Test constraints
        embed_num_confs = n_confs if not single_conf else 1
        try:
            mol = embed_func(mol, embed_num_confs)
        except Exception as e:
            print(e.output)
            pass
        if len(mol.GetConformers()) != embed_num_confs: # The number of conformers must match the specified number of confs
            print(len(mol.GetConformers()), '!=', embed_num_confs)
            return [], None
        if mmff: try_mmff(mol)

    if pdb: pdb = PDBFile(mol)
    conformers = []
    for i in range(n_confs): #
        data_conf = copy.deepcopy(data) # Copy data object: data_conf <--- data
        '''
        data_conf: # copied from data, AND ADD
            pos: position of a conformer
            seed_mol: an rdchem.Mol
        '''
        if single_conf: # False
            seed_mol = copy.deepcopy(mol)
        elif seed_confs: # None
            seed_mol = random.choice(seed_confs[smi])
        else:
            seed_mol = copy.deepcopy(mol) # Copy mol object: seed_mol <--- mol, seed_mol has n_confs conformers
            [seed_mol.RemoveConformer(j) for j in range(n_confs) if j != i] # Delete all confermers, excluding the i_th one.

        data_conf.pos = torch.from_numpy(seed_mol.GetConformers()[0].GetPositions()).float() # Add positions of the first conformer to $data_conf.pos
        data_conf.seed_mol = copy.deepcopy(seed_mol) # Add $seed_mol
        if pdb:
            pdb.add(data_conf.pos, part=i, order=0, repeat=still_frames)
            if seed_confs:
                pdb.add(data_conf.pos, part=i, order=-2, repeat=still_frames)
            pdb.add(torch.zeros_like(data_conf.pos), part=i, order=-1)

        conformers.append(data_conf)
    if mol.GetNumConformers() > 1:
        [mol.RemoveConformer(j) for j in range(n_confs) if j != 0]
    return conformers, pdb


def perturb_seeds(data, pdb=None):
    '''
    Sample uniform torsion angle
    Argument:
        data: A list of Data objects
    '''
    for i, data_conf in enumerate(data):
        torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
        # Apply uniform torsion angle
        data_conf.pos = modify_conformer(data_conf.pos, data_conf.edge_index.T[data_conf.edge_mask], # data_conf.edge_index.T[data_conf.edge_mask]: rotatable bonds/edges
                                         data_conf.mask_rotate, torsion_updates)
        data_conf.total_perturb = torsion_updates
        if pdb:
            pdb.add(data_conf.pos, part=i, order=1, repeat=still_frames)
    return data


def sample(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None):
    '''
    steps: Number of inference steps used by the resampler
    '''
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)
    '''
    DataBatch(x=[608, 44], edge_index=[2, 1152], edge_attr=[1152, 4], z=[608], name=[32], edge_mask=[1152], mask_rotate=[32], pos=[608, 3], seed_mol=[32], total_perturb=[32], idx=[32], batch=[608], ptr=[33])
        x: [no_nodes, no_features]
        edge_index: [2, no_edges] # Directional. Ex: [[1, 0], [0, 1]]
        edge_attr: [no_edges, 4]
        z: [no_nodes]
        name: [no_mols]
        edge_mask: [no_edges]
        mask_rotate: [no_mols] # list
        pos: [no_nodes, 3], 
        seed_mol: [no_mols], 
        total_perturb: [no_mols], # Sample angle at time T
        idx: [no_mols], 
        batch: [no_nodes] # A batch contains several(32) graphs.
                        The attribute batch separates atoms in the same graph.
        ptr: [33] # Batch/Graph separator
    '''
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1] # Dispose of the last element
    eps = 1 / steps

    for batch_idx, data in enumerate(loader):

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):

            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)
            
            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min))) # (Inference procedure Page 17)
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape) # Gaussian noise
            score = data_gpu.edge_pred.cpu() # Score

            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood)
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z # Page 17

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)

            if pdb:
                for conf_idx in range(data.num_graphs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()

    return conformers


def pyg_to_mol(mol, data, mmff=False, rmsd=True, copy=True):
    if not mol.GetNumConformers(): # If there is no confs.
        conformer = Chem.Conformer(mol.GetNumAtoms()) # The class to store 2D or 3D conformation of a molecule
        mol.AddConformer(conformer)
    coords = data.pos
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()

    # Transfer Atom Position from $data to $mol
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s') # Uses MMFF to optimize all of a molecule's conformations
        except Exception as e:
            pass
    try:
        if rmsd:
            mol.rmsd = AllChem.GetBestRMS( # Returns the optimal RMS for aligning two molecules
                Chem.RemoveHs(data.seed_mol),
                Chem.RemoveHs(mol)
            )
        mol.total_perturb = data.total_perturb
    except:
        pass
    mol.n_rotable_bonds = data.edge_mask.sum()
    if not copy: return mol
    return deepcopy(mol)


class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        for i, d in enumerate(data_list):
            d.idx = i
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def apply_torsion_and_update_pos(self, data, torsion_updates):

        pos_new, torsion_updates = perturb_batch(data, torsion_updates, split=True, return_updates=True)
        for i, idx in enumerate(data.idx):
            try:
                self.data[idx].total_perturb += torsion_updates[i]
            except:
                pass
            self.data[idx].pos = pos_new[i]
