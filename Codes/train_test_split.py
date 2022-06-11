import os
import random
import shutil

directory = 'train'
names = []

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f): 
	    if "jpg" in f: names.append(f)

indices = []
for i in range(300):
    indices.append(random.randint(0, 1499))

final = list(set(indices))
test_imgs = []
test_annots = []

for i in final:
    test_imgs.append(names[i].split('/')[-1])
    test_annots.append(names[i].split('/')[-1][:-3] + "xml")

    x = shutil.move("/home/hitesh/Downloads/main_DeepPCB/train/" + names[i].split('/')[-1],
                    "/home/hitesh/Downloads/main_DeepPCB/test/" + names[i].split('/')[-1])
    y = shutil.move("/home/hitesh/Downloads/main_DeepPCB/train/" + names[i].split('/')[-1][:-3] + "xml",
                    "/home/hitesh/Downloads/main_DeepPCB/test/" + names[i].split('/')[-1][:-3] + "xml")

print(test_imgs)
print(test_annots)
print(len(test_imgs))
