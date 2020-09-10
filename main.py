######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import imutils, argparse


from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load the trained convolutional neural network
print("[INFO] loading Network...")
model = load_model("models/NumberPlate.model")

LABELS = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H",
              "I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
def createPath(path_str):
    if not os.path.exists(path_str):
        os.makedirs(path_str)
createPath("detected")
def detectObject(image):
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (50, 100))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    model_val = model.predict(image)

    return model_val

# on the image I'm using, the headlamps were categorized as a license plate
# because their shapes were similar
# for now I'll just use the plate_like_objects[2] since I know that's the
# license plate. We'll fix this later

# The invert was done so as to convert the black pixel to white pixel and vice versa
cImg = 0
def getAndGet(image):
    global cImg
    #image = cv2.imread(DATA_DIR + imageName, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    license_plate = np.invert(im_bw)

    labelled_plate = measure.label(license_plate)
    
    character_dimensions = (0.15*license_plate.shape[0], 0.80*license_plate.shape[0], 0.01*license_plate.shape[1], 0.19*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    characters = []
    counter=0
    nImg = image.copy()
    nImg2 = image.copy()
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0
        
        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roiN = nImg[y0:y1, x0:x1]
            roiN2 = nImg2[y0:y1, x0:x1]
            res1 = detectObject(roiN)
            res = np.argmax(res1)
            if res == 36:
                pass
            else:
                if res1[0][res] > 0.7:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(nImg, str(LABELS[res]), (x0,y0-5), font, 1, (0,255,0), 2, cv2.LINE_AA)
                    cv2.rectangle(nImg, (x0,y0), (x1, y1), (255, 255, 0), 2)
            cImg += 1
    return nImg


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

def LoadModel(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES):
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Hre we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, sess, category_index



PATH_TO_CKPT = 'models/coco/frozen_inference_graph.pb'
PATH_TO_LABELS = 'models/coco/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
car_image_tensor, car_detection_boxes, car_detection_scores, car_detection_classes, car_num_detections, car_sess, car_category_index = LoadModel(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES)

PATH_TO_CKPT = 'models/frozen_inference_graph.pb'
PATH_TO_LABELS = 'models/labelmap.pbtxt'
NUM_CLASSES = 1
n_image_tensor, n_detection_boxes, n_detection_scores, n_detection_classes, n_num_detections, n_sess, n_category_index = LoadModel(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES)



counter = 0

def createDir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def detectAndSave(image, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, sess, category_index, modelToDetect="car"):
    global counter
    image = cv2.resize(image, (1366, 768))
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    boxesT = np.squeeze(boxes)
    
    

    # Get all the results which meet the given threshold
    thres = []
    for score in np.squeeze(scores):
        if score > 0.8:
            thres.append(score)
    thresClassesNum = len(thres)
    # All the results have been drawn on image. Now display the image.
    #cv2.imshow('Object detector', image)

    # Get image details
    width, height, depth = image.shape

    # Get unique Classes
    uniqueDirs = np.unique(np.squeeze(classes))

    # Get classes names and store in catName list
    catName = []
    catName.append("random")
    for key,val in category_index.items():
        for k, v in val.items():
            if k == "name":
                catName.append(v)

    createDir(P_IMG_PATH + PRED_DIR_NAME)
    # Create dirs if not exits
    try:
        for udir in uniqueDirs:
            createDir(P_IMG_PATH + PRED_OBJECT_DIR_NAME +str(catName[int(udir)]))
    except:
        print("Error")

    cnt = 0
    for i in range(0, thresClassesNum):
        cnt += 1
        ymin = int(boxesT[i][0] * width)
        xmin = int(boxesT[i][1] * height)
        ymax = int(boxesT[i][2] * width)
        xmax = int(boxesT[i][3] * height)

        # Slice the object
        objectImg = image[ymin:ymax, xmin: xmax]
        if modelToDetect == "n":
            if cnt > 0:
                img = getAndGet(objectImg)
                cv2.imshow("image", img)
        if modelToDetect == "car":
            print("Printing number palte")
            detectAndSave(objectImg, n_image_tensor, n_detection_boxes, n_detection_scores, n_detection_classes, n_num_detections, n_sess, n_category_index, modelToDetect="n")
        else:
            print("Going other way")
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    if modelToDetect == "n":
        if cnt > 0:
            cv2.imshow("img", image)
    

imag = cv2.imread("3img.jpg")
detectAndSave(imag, car_image_tensor, car_detection_boxes, car_detection_scores, car_detection_classes, car_num_detections, car_sess, car_category_index)

