import tensorflow as tf
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import shutil

import matplotlib.pyplot as plt
PATH_TO_SAVED_MODEL="/home/hitesh/Desktop/fsm_ml_trial/inference_graph_1/saved_model"
print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("/home/hitesh/Desktop/fsm_ml_trial/label_map.pbtxt",use_display_name=True)
#category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def get_tensor(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = input_tensor[..., tf.newaxis]
    input_tensor = tf.concat([input_tensor,input_tensor[:,:,:,:],
                          input_tensor[:,:,:,:]], 3)
    return input_tensor
    #return tf.concat([input_tensor[:,:,:,-1,:],input_tensor[:,:,:,-1,:], input_tensor[:,:,:,-1,:]], 3)          
    
def detect_it(image_path):
    image_np = np.asarray(np.array(Image.open(image_path)))
   
    detections = detect_fn(get_tensor(image_np))
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    output = []

    for i in range(len(detections['detection_scores'])):
        if detections['detection_scores'][i]>0.4:
            lis = []
            lis.append(label_map[detections['detection_classes'][i]])
            lis.append(detections['detection_scores'][i])
            positions = [int(640*m) for m in detections['detection_boxes'][i]]
            lis.append(positions[1])
            lis.append(positions[0])
            lis.append(positions[3])
            lis.append(positions[2])
            output.append(lis)
    
    return output
    
directory = '/home/hitesh/Desktop/test_imgs'
names = []

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f): names.append(f)
	

label_map = ['background', 'copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
count = 0
    
for k in names:
    image_path = k
    output = detect_it(image_path)
    
    x = shutil.move(k, "/home/hitesh/Desktop/done_imgs/" + k.split('/')[-1])        
                    
    with open("/home/hitesh/Desktop/test_preds/" + k.split('/')[-1][:-3] + "txt", "w") as f:
        for j in output:
            for l in j:
                f.write(str(l))
                f.write(" ")
            f.write("\n")
    count = count + 1
    print(str(count) + " - " + k)
    
    
print(len(names))
