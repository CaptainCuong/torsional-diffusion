import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from copy import deepcopy
import os
from rdkit.Chem import Draw
from matplotlib.colors import ColorConverter

parser = ArgumentParser()
parser.add_argument('--confs', type=str, default='test_run/test_data.pkl', help='Path to pickle file with generated conformers')
parser.add_argument('--stat_report', type=bool, default=True, help='Export statistical reports')
parser.add_argument('--test_csv', type=str, default='./data/QM9/test_smiles.csv', help='Path to csv file with list of smiles')
parser.add_argument('--true_mols', type=str, default='D:/Github/torsional-diffusion/data/QM9/test_mols.pkl', help='Path to pickle file with ground truth conformers')
parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
parser.add_argument('--dataset', type=str, default="qm9", help='Dataset: drugs, qm9 and xl')
parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
args = parser.parse_args()

"""
    Evaluates the RMSD of some generated conformers w.r.t. the given set of ground truth
    Part of the code taken from GeoMol https://github.com/PattanaikL/GeoMol
"""

########################## LOAD GENERATED CONFS
with open(args.confs, 'rb') as f:
    model_preds = pickle.load(f) # 931 groundtruth molecules for testing

test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
'''
13731 conformers
                      smiles  n_conformers         corrected_smiles
0         C#CC#C[C@@H](CC)CO            29       C#CC#C[C@@H](CC)CO
1    C#CC#C[C@H](O)[C@H]1CN1            10  C#CC#C[C@H](O)[C@H]1CN1
2              C#CC(=O)CCCCC            58            C#CC(=O)CCCCC
3       C#CC(=O)C[C@H](O)C#C            21     C#CC(=O)C[C@H](O)C#C
'''
##########################

# LOAD GROUNDTRUTH CONFS
with open(args.true_mols, 'rb') as f:
    true_mols = pickle.load(f)
threshold = threshold_ranges = np.arange(0, 2.5, .125)
'''
rdkit_smiles: extract mol from $true_mols
corrected_smiles: used for the function $clean_confs
    model_preds also uses this smile
'''

def calc_performance_stats(rmsd_array):
    '''
    res in results.values():
    rmsd_array <-- res['rmsd'] : np.array(n_true, n_model)
    '''
    coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0) # List
    amr_recall = rmsd_array.min(axis=1).mean() # Scalar
    coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1) # List
    amr_precision = rmsd_array.min(axis=0).mean() # Scalar

    return coverage_recall, amr_recall, coverage_precision, amr_precision

def calc_performance_stats_const_threshold(rmsd_array, const_threshold = 0.5):
    '''
    res in results.values():
    rmsd_array <-- res['rmsd'] : np.array(n_true, n_model)
    '''
    coverage_recall = np.mean(rmsd_array.min(axis=1) < const_threshold) # Scalar
    amr_recall = rmsd_array.min(axis=1).mean() # Scalar
    coverage_precision = np.mean(rmsd_array.min(axis=0) < const_threshold) # Scalar
    amr_precision = rmsd_array.min(axis=0).mean() # Scalar

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def clean_confs(smi, confs):
    '''
    Clean valid groundtruth confs
    '''
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    '''
    Ex: smi: C#CC#CC(CC)CO
    MolFromSmiles: Construct a molecule from a SMILES string
        sanitize: (optional) toggles sanitization of the molecule. Defaults to True.

    MolToSmiles: Returns the canonical SMILES string for a molecule
        isomericSmiles: (optional) include information about stereochemistry in the SMILES. Defaults to true.
        (cis-trans)
    '''
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i) # Append index of valid groundtruth confs
    return [confs[i] for i in good_ids]

########################## LOAD SMILES
rdkit_smiles = test_data.smiles.values
# ['C#CC#C[C@@H](CC)CO', 'C#CC#C[C@H](O)[C@H]1CN1', ...]
corrected_smiles = test_data.corrected_smiles.values
# ['C#CC#C[C@@H](CC)CO', 'C#CC#C[C@H](O)[C@H]1CN1', ...]
if args.limit_mols:
    rdkit_smiles = rdkit_smiles[:args.limit_mols]
    corrected_smiles = corrected_smiles[:args.limit_mols]
##########################

num_failures = 0
results = {}
jobs = []
'''
jobs: List((smi, corrected_smi, i_true))
    i_true: index for $n_true in $results[(smi, corrected_smi)]

results: {
    (smi, corrected_smi): {
        'n_true': n_true, # number of true confs
        'n_model': n_model, # number of generated confs
        'rmsd': np.nan * np.ones((n_true, n_model))
    }
}
'''

filter_mols = None
if args.filter_mols:
    with open(args.filter_mols, 'rb') as f:
        filter_mols = pickle.load(f)

for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):

    if filter_mols is not None and corrected_smi not in filter_mols:
        continue

    if args.dataset == 'xl':
        smi = corrected_smi

    if corrected_smi not in model_preds:
        print('model failure', corrected_smi)
        num_failures += 1
        continue

    true_mols[smi] = true_confs = clean_confs(corrected_smi, true_mols[smi]) # clean valid groundtruth confs (conf of true_mols[smi] with smile == corrected_smi )
    
    ############ Add to visualization folder ############
    if not os.path.exists(f'visualization/{smi}'):
        os.makedirs(f'visualization/{smi}')
    img = Chem.MolFromSmiles(smi)
    img = Chem.Draw.MolToImage(img, highlightAtoms=[1,2], highlightColor=ColorConverter().to_rgb('aqua'))
    img.save(f'visualization/{smi}/mol_img.png')
    if corrected_smi in model_preds:
        [Chem.rdmolfiles.MolToPDBFile(model_preds[corrected_smi][i], f'visualization/{smi}/conf_{i}.pdb') for i in range(len(model_preds[corrected_smi]))]
        [Chem.rdmolfiles.MolToPDBFile(true_mols[smi][i], f'visualization/{smi}/std_{i}.pdb') for i in range(len(true_mols[smi]))]
    ############################################################

    if len(true_confs) == 0:
        print(f'poor ground truth conformers: {corrected_smi}')
        continue

    n_true = len(true_confs)
    n_model = len(model_preds[corrected_smi])
    results[(smi, corrected_smi)] = {
        'n_true': n_true, # number of true confs
        'n_model': n_model, # number of generated confs
        'rmsd': np.nan * np.ones((n_true, n_model))
    }
    for i_true in range(n_true):
        jobs.append((smi, corrected_smi, i_true))


def worker_fn(job):
    r'''
    Add $rmsds - list of RMSDs between a true conf and generated confs with $correct_smi

    Parameter:
        job (Tuple): (smi, corrected_smi, i_true)
            Ex: ('C#CC#C[C@@H](CC)CO', 'C#CC#C[C@@H](CC)CO', 25)
    '''
    smi, correct_smi, i_true = job
    true_confs = true_mols[smi]
    model_confs = model_preds[correct_smi]
    tc = true_confs[i_true]
    rmsds = []

    for mc in model_confs:
        try:
            if args.only_alignmol:
                rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem. veHs(mc))
            else:
                rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            rmsds.append(rmsd)
        except:
            print('Additional failure', smi, correct_smi)
            rmsds = [np.nan] * len(model_confs)
            break
    return smi, correct_smi, i_true, rmsds


def populate_results(res):
    '''
    Add $rmsd of each generated conf wrt a groundtruth conf into $results
    '''
    smi, correct_smi, i_true, rmsds = res
    results[(smi, correct_smi)]['rmsd'][i_true] = rmsds


random.shuffle(jobs)
if args.n_workers > 1:
    p = Pool(args.n_workers)
    map_fn = p.imap_unordered
    p.__enter__()
else:
    map_fn = map

# Calculate $rmsd and add it into $results
for res in tqdm(map_fn(worker_fn, jobs), total=len(jobs)):
    populate_results(res)

#################### STAT REPORT ####################
if args.stat_report:
    temp_res = []
    for key, val in results.items():
        temp_dict = deepcopy(val)
        temp_dict['smi'] = key[0]
        temp_dict['rmsd'] = calc_performance_stats_const_threshold(val['rmsd'], const_threshold = 0.5)
        temp_res.append(temp_dict)

    def sort_cov_r(dict):
        return -dict['rmsd'][0]

    def sort_amr_r(dict):
        return dict['rmsd'][1] 

    def sort_cov_p(dict):
        return -dict['rmsd'][2] 

    def sort_amr_p(dict):
        return dict['rmsd'][3] 

    report = open('sort_cov_r.csv', 'w')
    temp_res.sort(key=sort_cov_r)
    for idx, x in enumerate(temp_res):
        report.write(f"{idx+1},{x['smi']},{x['rmsd'][0]}\n")
    report.close()

    report = open('sort_amr_r.csv', 'w')
    temp_res.sort(key=sort_amr_r)
    for idx, x in enumerate(temp_res):
        report.write(f"{idx+1},{x['smi']},{x['rmsd'][1]}\n")
    report.close()

    report = open('sort_cov_p.csv', 'w')
    temp_res.sort(key=sort_cov_p)
    for idx, x in enumerate(temp_res):
        report.write(f"{idx+1},{x['smi']},{x['rmsd'][2]}\n")
    report.close()

    report = open('sort_amr_p.csv', 'w')
    temp_res.sort(key=sort_amr_p)
    for idx, x in enumerate(temp_res):
        report.write(f"{idx+1},{x['smi']},{x['rmsd'][3]}\n")
    report.close()
############################################################

# if args.n_workers > 1:
#     p.__exit__(None, None, None)

# stats = []
# for res in results.values():
#     stats_ = calc_performance_stats(res['rmsd'])
#     cr, mr, cp, mp = stats_
#     stats.append(stats_)
# coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

# for i, thresh in enumerate(threshold_ranges):
#     print('threshold', thresh)
#     coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * num_failures
#     coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * num_failures
#     print(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
#     print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
#     print(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
#     print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')


# report = open('report.csv','w')
# report.write('threshold,Recall Coverage-Mean,Recall Coverage-Median,Recall AMR-Mean,Recall AMR-Median,Precision Coverage-Mean,Precision Coverage-Median,Precision AMR-Mean,Precision AMR-Median\n')

# for i, thresh in enumerate(threshold_ranges):
#     coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * num_failures
#     coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * num_failures
#     report.write(f'{thresh},\
#         {np.mean(coverage_recall_vals) * 100:.2f},{np.median(coverage_recall_vals) * 100:.2f},\
#         {np.nanmean(amr_recall):.4f},{np.nanmedian(amr_recall):.4f},\
#         {np.mean(coverage_precision_vals) * 100:.2f},{np.median(coverage_precision_vals) * 100:.2f},\
#         {np.nanmean(amr_precision):.4f},{np.nanmedian(amr_precision):.4f}\n')
# report.close()


# print(len(results), 'conformer sets compared', num_failures, 'model failures', np.isnan(amr_recall).sum(),
#       'additional failures')
