#Imports
import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
  
os.chdir('D:/deep learning/models-master/research/object_detection')
  
#Env setup 
# This is needed to display the images.
#%matplotlib inline
 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
 
#Object detection imports
from utils import label_map_util
 
from utils import visualization_utils as vis_util
 
 
#Model preparation
# What model to download.
 
#这是我们刚才训练的模型
#MODEL_NAME = 'D:/1deep learning/models'
 
#对应的Frozen model位置
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'D:/deep learning/pbfiles/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('D:/deep learning/train/', 'label_map.pbtxt')
 
#改成自己例子中的类别数，4
NUM_CLASSES = 1
 
#Load a (frozen) Tensorflow model into memory.    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')    
    
#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
 
#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
 
 
#Detection
 
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#测试图片位置
PATH_TO_TEST_IMAGES_DIR = 'D:/deep learning/train/pic'
os.chdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
 
# Size, in inches, of the output images.
IMAGE_SIZE = (1024, 1024)
 
output_path = ('D:/deep learning/eval_pic_60/')
 
total_times = 0
w = 0
h = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            
            image = cv2.imread(image_path, 0)
            w , h = image.shape
            image_RGB = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image_np = image_RGB
        
            image_np_expanded = np.expand_dims(image_np, axis=0)

            start = time.time()
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
            end =  time.time()
            print("once infer spend times: ",end -start)
            total_times += (end - start)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
        
            cv2.imwrite(output_path + image_path.split('\\')[-1],image_np)	
      
print("Picture size: ", w , h )
print(" Picture nums: ", len(TEST_IMAGE_PATHS)," Total spend times: ",total_times)
