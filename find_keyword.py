import os

file_dir = ['./diffusion', './utils']
keyword = '.*cpu.*'

for dir_ in file_dir:
	os.system('wsl -e sh -c "grep -worne \'' + keyword + '\' ' + dir_ + '"')
	print('\n','-'*50,'\n')

os.system('wsl -e sh -c "grep -wone \'' + keyword + '\' ' + '* .*"')
