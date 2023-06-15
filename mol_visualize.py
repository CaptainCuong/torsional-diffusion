from rdkit.Chem import Draw
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem
import os
from matplotlib.colors import ColorConverter 

data = pickle.load(open('test_run/test_data_tree.pkl', 'rb'))
standard_data = pickle.load(open('data/QM9/test_mols.pkl', 'rb'))

for key, mols in list(data.items()):
	if key not in standard_data:
		continue
	if not os.path.exists(f'visualization_tree/{key}'):
		os.makedirs(f'visualization_tree/{key}')
	img = Chem.MolFromSmiles(key)
	img = Chem.Draw.MolToImage(img, highlightAtoms=[1,2], highlightColor=ColorConverter().to_rgb('aqua'))
	img.save(f'visualization_tree/{key}/mol_img.png')
	for i in range(len(mols)):
		Chem.rdmolfiles.MolToPDBFile(mols[i], f'visualization_tree/{key}/conf_{i}.pdb')
	for i in range(len(standard_data[key])):
		Chem.rdmolfiles.MolToPDBFile(standard_data[key][i], f'visualization_tree/{key}/std_{i}.pdb')