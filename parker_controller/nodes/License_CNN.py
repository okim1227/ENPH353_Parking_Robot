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
		img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		cv_img = cv2.resize(img, (175, 41), interpolation = cv2.INTER_AREA)
		# cv2.imshow("pic", cv_img)
		# cv2.waitKey(0)
		gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
		image, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cv2.waitKey(0)
		cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
		last_index = len(cntsSorted) - 1
		if(cv2.contourArea(cntsSorted[last_index]) > 50):
			x_points = []
			img = []
			for cnt in contours:
				x,y,w,h = cv2.boundingRect(cnt)
				x_points.append(x)
			x_points.sort()
			img.append(binary[:,x_points[0]:x_points[0] + 25])
			img.append(binary[:,x_points[0] + 28: x_points[0] + 53])
			img.append( binary[:,x_points[0] + 86: x_points[0] + 111])
			img.append(binary[:,x_points[0] + 113: x_points[0] + 138])

			for i in range(0,4):
				image = img[i]
				h,w=image.shape[0:2]
				print(h)
				print(w)
				M = cv2.moments(image)
				M["m00"] != 0
				cX = int(M["m10"] / M["m00"]) 
				cY = int(M["m01"] / M["m00"])
				Ydiff = (cY - 20) / 2
				Xdiff = (cX - 12) / 2
				if (Ydiff < 0):
					bottom = 0
					top = abs(Ydiff)
					image = image[0:h+Ydiff,:]
				elif(Ydiff > 0):
					bottom = Ydiff
					top = 0
					image = image[Ydiff:h,:]

				if (Xdiff < 0):
					right = 0
					left = abs(Xdiff)
					image = image[:,0:w+Xdiff]
				elif(Xdiff > 0):
					right = Xdiff
					left = 0
					image = image[:,Xdiff:w]

				image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				img[i] = cv2.resize(image, (25, 41), interpolation = cv2.INTER_AREA)
				print(img[i].shape)


			plt.imshow(img[0], cmap='gray')
			plt.show()
			plt.imshow(img[1], cmap='gray')
			plt.show()
			plt.imshow(img[2], cmap='gray')
			plt.show()
			plt.imshow(img[3], cmap='gray')
			plt.show()
		cv2.destroyAllWindows()




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
