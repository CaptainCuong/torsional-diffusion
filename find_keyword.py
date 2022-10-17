# import os

# file_dir = ['./diffusion', './utils']
# keyword = '.*TorsionNoiseTransform.*'

# for dir_ in file_dir:
# 	os.system('wsl -e sh -c "grep -worne \'' + keyword + '\' ' + dir_ + '"')
# 	print('\n','-'*50,'\n')

# os.system('wsl -e sh -c "grep -wone \'' + keyword + '\' ' + '* .*"')

from utils.parsing import parse_train_args
from utils.dataset import construct_loader
import math, os, torch, yaml, tqdm

if __name__ == '__main__':
    args = parse_train_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = construct_loader(args)
    print(len(train_loader))
    for data in tqdm.tqdm(train_loader, total=len(train_loader)):
    	print(data)