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

# sess1 = tf.Session()    
# graph1 = tf.get_default_graph()
# set_session(sess1)
# plate_NN = models.load_model("/home/fizzer/ros_ws/src/parker_controller/nodes/my_model.h5")

class license_cnn:

	def __init__(self):
		self.bridge = CvBridge()
		self.plate_sub = rospy.Subscriber("/license_cnn", Image,self.plate_callback)
		time.sleep(1)

	def plate_callback(self, data):
		#process the images then send it to perdict
		cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		cv2.waitKey(0)
		gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
		image, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
		last_index = len(cntsSorted) - 1
		# img1 = binary[:, start_point: start_point + 15]
		# img2 = binary[:, start_point+15: start_point + 30]
		# img3 = binary[:, start_point+50: start_point + 65]
		# img4 = binary[:, start_point+80: start_point + 95]
		x,y,w,h = cv2.boundingRect(cntsSorted[last_index])
		cv2.rectangle(cv_img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow("num", cv_img)
		cv2.waitKey(0)
		# plt.imshow(img1, cmap='gray')
		# plt.show()
		# plt.imshow(img2, cmap='gray')
		# plt.show()
		# plt.imshow(img3, cmap='gray')
		# plt.show()
		# plt.imshow(img4, cmap='gray')
		# plt.show()




	def predict(image):
		global sess1
		global graph1
		with graph1.as_default():
			set_session(sess1)
			NN_prediction = plate_NN.predict(image)[0]
		return NN_prediction

	#blank_image = np.array([np.zeros((125, 105, 3))])

	#print(predict(blank_image))

def main(args):
	rospy.init_node("license_cnn", anonymous=True)
	ic = license_cnn()
	rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
