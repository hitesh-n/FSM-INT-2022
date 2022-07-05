import tensorflow as tf
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageDraw, ImageFont 
import os
import shutil
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from models.research.object_detection.utils import label_map_util

label_map = ['background', 'copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
PATH_TO_SAVED_MODEL="/home/hitesh/Desktop/flask_deploy_trial/frcnn_model/inference_graph_frcnn/saved_model"
print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("/home/hitesh/Desktop/flask_deploy_trial/frcnn_model/label_map.pbtxt",use_display_name=True)
#category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)

def get_tensor(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = input_tensor[..., tf.newaxis]
    input_tensor = tf.concat([input_tensor,input_tensor[:,:,:,:],
                          input_tensor[:,:,:,:]], 3)
    return input_tensor
    #return tf.concat([input_tensor[:,:,:,-1,:],input_tensor[:,:,:,-1,:], input_tensor[:,:,:,-1,:]], 3)          
    
def detect_defects(image_path):
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
    
 
#*** Backend operation
 
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticfiles', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask('__name__', template_folder='templates', static_folder='staticfiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')
 
@app.route('/',  methods=["POST", "GET"])
def uploadFile():

    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        return render_template('index_upload_and_show_data_page2.html', user_image = session.get('uploaded_img_file_path', None))

@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    output = detect_defects(img_file_path)
    img = Image.open(img_file_path)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    # choose from here /usr/share/fonts/truetype/freefont
    myFont = ImageFont.truetype('FreeMonoBold.ttf', 15)
    for i in output:
        draw.rectangle([(i[2],i[3]),(i[4],i[5])], outline="red", width=5)
        draw.text((i[2], i[3]-20), i[0], fill ="yellow", font = myFont)
    img.save("staticfiles/xyz.jpg")    
    # Display image in Flask application web page
    return render_template('show_image.html', user_image = "staticfiles/xyz.jpg")
 
if __name__=='__main__':
    app.run(debug = True)

