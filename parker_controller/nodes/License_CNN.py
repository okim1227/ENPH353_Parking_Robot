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

import os

directory = r'/home/fizzer/Desktop/License_Images'
os.chdir(directory)

	
sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)
plate_NN = models.load_model("/home/fizzer/ros_ws/src/parker_controller/nodes/licence_CNN_model.h5")

class license_cnn:

	def __init__(self):
		self.bridge = CvBridge()
		self.i = 2
		self.license_pub = rospy.Publisher("/license_plate", String, queue_size=1)
		self.plate_sub = rospy.Subscriber("/license_cnn", Image,self.plate_callback)
		self.array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
		time.sleep(1)

	def plate_callback(self, data):
		cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		# cv2.imshow("pic", cv_img)
		# cv2.waitKey(0)
		gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
		image, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		# cv2.imshow("pic",binary)
		# cv2.waitKey(0)
		cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
		last_index = len(cntsSorted) - 1
		if(last_index >= 3 and cv2.contourArea(cntsSorted[last_index]) > 28):
			print("Recieved")
			position = []
			for i in range(0,4):
				x,y,w,h = cv2.boundingRect(cntsSorted[last_index - i])
				position.append([x,y])
			positionSorted = sorted(position, key=lambda x: x[0])
			img1 = binary[int(positionSorted[0][1]):int(positionSorted[0][1])+25,positionSorted[0][0]:positionSorted[0][0]+25]
			img2 = binary[int(positionSorted[1][1]):int(positionSorted[1][1])+25,positionSorted[1][0]:positionSorted[1][0]+25]
			img3 = binary[int(positionSorted[2][1]):int(positionSorted[2][1])+25,positionSorted[2][0]:positionSorted[2][0]+25]
			img4 = binary[int(positionSorted[3][1]):int(positionSorted[3][1])+25,positionSorted[3][0]:positionSorted[3][0]+25]
			img1 = cv2.resize(img1, (25,25), interpolation = cv2.INTER_AREA)
			img2 = cv2.resize(img2, (25,25), interpolation = cv2.INTER_AREA)
			img3 = cv2.resize(img3, (25,25), interpolation = cv2.INTER_AREA)
			img4 = cv2.resize(img4, (25,25), interpolation = cv2.INTER_AREA)
			image_final_1 = cv2.merge((img1,img1,img1))
			image_final_2 = cv2.merge((img2,img2,img2))
			image_final_3 = cv2.merge((img3,img3,img3))
			image_final_4 = cv2.merge((img4,img4,img4))
			index_1 = np.argmax(self.predict(np.array([image_final_1])))
			index_2 = np.argmax(self.predict(np.array([image_final_2])))
			index_3 = np.argmax(self.predict(np.array([image_final_3])))
			index_4 = np.argmax(self.predict(np.array([image_final_4])))
			print(self.array[index_1] + self.array[index_2] + self.array[index_3] + self.array[index_4] )
			TeamRed,multi21,4,XR58
			self.license_pub.publish("PARKER,multi12," +  str(self.i) + str(self.array[index_1]) + str(self.array[index_2]) + str(self.array[index_3]) + str(self.array[index_4]))
			self.i = -1
		cv2.destroyAllWindows()


	def predict(self, image):
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
