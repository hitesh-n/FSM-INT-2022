# code used to change file paths in xml files to appropriate directories for test 

import os

directory = 'test_data'
names = []

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f): 
	    if "xml" in f: names.append(f)

for i in names:

	with open(i) as f:
	    lines = f.readlines()
	    
	lines[3] = "<path>./drive/MyDrive/trial_DeepPCB/test_data/" + i.split("/")[-1][:-3] + "jpg</path>"
	
	with open(i, "w") as f: 
	    f.writelines(lines)

