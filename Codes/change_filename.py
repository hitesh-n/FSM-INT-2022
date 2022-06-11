# this code is used to modify the annotation xml files to match the path of current system 

import os

directory = 'train'
names = []

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f): names.append(f)

for i in names:
	if "xml" in i: 
		with open(i) as file: data = file.readlines()
    		# modifies the location of the image file to match existing machine
		data[3] = "<path>./drive/MyDrive/trial/train/" + i.split('/')[-1][:-3] + "jpg</path>" 
		with open(i, "w") as file: file.writelines(data)
	
print(names)
