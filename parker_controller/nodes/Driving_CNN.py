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


sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)
driving_NN = models.load_model("/home/fizzer/ros_ws/src/parker_controller/nodes/driving_CNN_model.h5")


class driving_cnn:

	def __init__(self):
		self.bridge = CvBridge()
		self.cmdVel_pub = rospy.Publisher("/R1/cmd_vel", Twist)
		self.camera_sub = rospy.Subscriber("/R1/camera1/image_raw",Image,self.callback)
		time.sleep(1)

	def callback(self, image):
		cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		message = self.predict(cv_img)
		self.cmdVel_pub.publish(message)

	
	def predict(slef, image):
		global sess1
		global graph1
		with graph1.as_default():
			set_session(sess1)
			NN_prediction = driving_NN.predict(image)[0]
		return NN_prediction

def main(args):
	rospy.init_node("driving_cnn", anonymous=True)
	ic = driving_cnn()
	rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
