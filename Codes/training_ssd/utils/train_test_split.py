import os
import random
import shutil

directory = 'train_labels'
names = []

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f): names.append(f)

indices = []
for i in range(300):
    indices.append(random.randint(0, 1499))

final = list(set(indices))
test_imgs = []
test_annots = []

for i in final:
    test_imgs.append(names[i].split('/')[-1])

    x = shutil.move("/home/hitesh/Desktop/fsm_ssd/train_labels/" + names[i].split('/')[-1],
                    "/home/hitesh/Desktop/fsm_ssd/test_labels/" + names[i].split('/')[-1])
    

print(test_imgs)
print(test_annots)
print(len(test_imgs))
