#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt 
import os
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import numpy as np

class license_cnn:

	def __init__(self):
		sess1 = tf.Session()    
		graph1 = tf.get_default_graph()
		set_session(sess1)
		#print(os.getcwd())
		self.plate_NN = models.load_model("/home/fizzer/ros_ws/src/parker_controller/nodes/my_model.h5")
		self.plate_sub = rospy.Subscriber("/license_cnn", Image,self.plate_callback)

	def plate_callback(self, data):
		#process the images then send it to perdict
		#cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		print("dog")


	# def predict(image):
	# 	global sess1
	# 	global graph1
	# 	with graph1.as_default():
	# 		set_session(sess1)
	# 		NN_prediction = plate_NN.predict(image)[0]
	# 	return NN_prediction

	#blank_image = np.array([np.zeros((125, 105, 3))])

	#print(predict(blank_image))

def main(args):
	rospy.init_node("license_cnn", anonymous=True)
	ic = license_cnn()
	rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
