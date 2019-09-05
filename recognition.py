#!/usr/bin/env python
#this code is inspired by https://github.com/simontomaskarlsson/deep_learning_gesture_recognition
#please check the link to see the original version
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from PIL import Image, ImageDraw

import glob
import os

import argparse
import sys
import time
import os
import random
from datetime import datetime

import tensorflow as tf

#Pepper robot
from naoqi import ALProxy
IP = "10.0.10.41" #IP of the robot
PORT = 9559 #Port of the robot

# pepper camera
try:
    pepper_camera = ALProxy("ALVideoDevice", IP, PORT)
except Exception, e:
    print("Could not create proxy to ALAudioPlayer")
    print ("Error was: ", e)
    sys.exit(1)


def subscribe_pepper_camera(camera, resolution, fps):
        """
        Example usage:
        >>> pepper_camera(0, 1, 15)
        :param camera: `camera_depth`, `camera_top` or `camera_bottom`
        :type camera: string
        :param resolution:
            0. 160x120
            1. 320x240
            2. 640x480
            3. 1280x960
        :type resolution: integer
        :param fps: Frames per sec (5, 10, 15 or 30)
        :type fps: integer
        """
        color_space = 13

        camera_index = None
        if camera == "camera_top":
            camera_index = 0
        elif camera == "camera_bottom":
            camera_index = 1
        elif camera == "camera_depth":
            camera_index = 2
            resolution = 1
            color_space = 11

        camera_link = pepper_camera.subscribeCamera("Camera_Stream" + str(np.random.random()),
                                                              camera_index, resolution, color_space, fps)
        if camera_link:
            print("[INFO]: Camera is initialized")
        else:
            print("[ERROR]: Camera is not initialized properly")

        image_raw = pepper_camera.getImageRemote(camera_link)
        image = np.frombuffer(image_raw[6], np.uint8).reshape(image_raw[1], image_raw[0], 3)

        return image

def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def run_graph(image_data, labels, input_layer_name, sess, softmax_tensor,
              num_top_predictions):

    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    print(labels[top_k[0]])
    #human_string = labels[top_k[0]]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return human_string, labels[top_k[0]]

def startWebcam(classificationFunc):

    waitingLimit = 30
    predictionText = ""
    myIndex = 0
    cv2.namedWindow("preview")
    rval = True # for Pepper
    frame = subscribe_pepper_camera("camera_top", 1, 30)

    while rval:

        waitingLimit -= 1
        frame = subscribe_pepper_camera("camera_top", 1, 30)
        if waitingLimit == 15:
            resizedFrame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
            resizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            resizedImage = Image.fromarray(resizedFrame, "RGB")
            resizedImage.save("/home/tensorflow-for-poets-2/scripts/webcam/temp.jpg")

        if waitingLimit == 0:
            resizedFrame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
            resizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            loaded_image = load_image("/home/tensorflow-for-poets-2/scripts/webcam/temp.jpg")
            predictions, top_label = classificationFunc(loaded_image)
            predictionText = "class: " + str(top_label) #flipping class

        waitingLimit = 30

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,predictionText,(0,50), font, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("preview", frame)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")



def main():
    # load labels
    labels = load_labels("/home/tensorflow-for-poets-2/tf_files/retrained_labels.txt")

    # load graph, which is stored in the default session
    load_graph("/home/tensorflow-for-poets-2/tf_files/retrained_graph.pb")

    with tf.Session() as sess:
       # Feed the image_data as input to the graph.
       #   predictions will contain a two-dimensional array, where one
       #   dimension represents the input image count, and the other has
       #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        startWebcam(lambda image: run_graph(image, labels, 'DecodeJpeg/contents:0', sess, softmax_tensor,
            5))

if __name__ == '__main__':
    main()

