# Utilities for object detector.

import numpy as np
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(
            num_hands_detect, 
            score_thresh, 
            scores, 
            boxes, 
            im_width, 
            im_height, 
            image_np, 
            visualize
            ):

    center_x = 0
    center_y = 0
    detected_boxes = [None]

    max_idx = np.argmax(scores)

    if (scores[max_idx] > score_thresh):
        detected_boxes = boxes[max_idx].tolist()
        (left, right, top, bottom) = (boxes[max_idx][1] * im_width, boxes[max_idx][3] * im_width,
                                      boxes[max_idx][0] * im_height, boxes[max_idx][2] * im_height)
        center_x = int((left+right)/2)
        center_y = int((top+bottom)/2)
        
        if visualize:
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.circle(image_np, (center_x, center_y), 1,(77, 255, 9), 2)
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
 
    return center_x, center_y, detected_boxes

def draw_ARbbox_on_image(
            boxes, 
            im_width, 
            im_height, 
            image_np, 
            isWashing,
            ):
    i = 0
    if boxes[0] != None:
        (left, right, top, bottom) = (boxes[1] * im_width, boxes[3] * im_width,
                                      boxes[0] * im_height, boxes[2] * im_height)
        cx = int((left+right)/2)
        cy = int((top+bottom)/2)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.circle(image_np, (cx, cy), 1,(77, 255, 9), 2)
        if isWashing:
            cv2.putText(image_np, "washing hand", (p1[0], p1[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
        else:
            cv2.putText(image_np, "Please wash properly", (p1[0], p1[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.rectangle(image_np, p1, p2, (255, 0, 0), 3, 1)


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

def draw_framecount_on_image(fc, image_np):
    cv2.putText(image_np, fc, (320, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
