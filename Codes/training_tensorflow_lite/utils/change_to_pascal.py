#code used to convert DeepPCB annotation to pascalVOC

import os

directory = 'train'
names = []

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f): names.append(f)

count = 0
defects =  ["background", "open", "short", "mousebite", "spur", "copper", "pin-hole"]
for i in names:
	if "txt" in i:
		with open(i) as f:
			lines = f.readlines()

		with open(i.split("/")[-1][:-4] + "_test.xml", "w") as f:
			f.write("<annotation>\n<folder>train</folder>\n<filename>" + i.split("/")[-1][:-4]+
					"_test.jpg</filename>\n<path>./drive/MyDrive/trial_DeepPCB/train/"
					+ i.split("/")[-1][:-4] + "_test.jpg</path>\n" +
					"<source> <database>Unknown</database> </source>\n" +
					"<size> <width>640</width> <height>640</height>\n" +
		     		"<depth>3</depth> </size>\n <segmented>0</segmented>\n")

			for j in lines:
				locations = j.split()
				f.write("<object> <name>" + defects[int(locations[-1])] +
						"</name> <pose>Unspecified</pose> " +
						"<truncated>0</truncated>\n" +
						"<difficult>0</difficult>\n<bndbox>\n" +
						"<xmin>" + locations[0] + "</xmin>\n" +
						"<ymin>" + locations[1] + "</ymin>\n" +
						"<xmax>" + locations[2] + "</xmax>\n" +
						"<ymax>" + locations[3] + "</ymax>\n</bndbox>\n</object>\n")

			f.write("</annotation>")
	count = count + 1
	print(count)
