from rdkit.Chem import Draw
import pickle
from rdkit.Chem import AllChem

file = open('data/QM9/standardized_pickles/000.pickle', 'rb')
file = pickle.load(file)

mol_names = list(file.keys())
for mol_name in mol_names:
	mol = file[mol_name]['conformers'][0]['rd_mol']
	print(AllChem.Compute2DCoords(mol))
	print(mol_name)
	Draw.MolToFile(mol,'data/mol_image/'+mol_name+'.png')    