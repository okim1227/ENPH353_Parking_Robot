#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt 
# import os

# directory = r'/home/fizzer/Desktop/License_Images'
# os.chdir(directory)

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

class license_extraction:

	def __init__(self):
		self.plate_pub = rospy.Publisher("license_cnn", Image)
		self.bridge = CvBridge()
		self.camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback)
		time.sleep(1)

	def camera_callback(self,data):
		cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
		cv2.waitKey(0)
		gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(gray, 70, 200, cv2.THRESH_BINARY_INV)
		image_copy = cv_img[400:550, 0:250]
		crop_img = binary[400:550, 0:250]
		image, contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
		last_index = len(cntsSorted) - 1
		a = cv2.drawContours(image=image_copy, contours=cntsSorted, contourIdx=last_index, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
		b = cv2.drawContours(image=image_copy, contours=cntsSorted, contourIdx=last_index-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

		if (last_index > 0 and cv2.contourArea(cntsSorted[last_index]) > 2500 and cv2.contourArea(cntsSorted[last_index -1 ]) > 2500):
			end_points_a = cv2.approxPolyDP(cntsSorted[last_index], 0.01 * cv2.arcLength(cntsSorted[last_index], True), True)
			end_points_b = cv2.approxPolyDP(cntsSorted[last_index - 1], 0.01 * cv2.arcLength(cntsSorted[last_index - 1], True), True)
			if (len(end_points_a) == 4 and len(end_points_b) == 4 and cv2.contourArea(cntsSorted[last_index]) == cv2.contourArea(cntsSorted[last_index]-1)):
				print("here")
				if (abs(end_points_a[0][0][1] - end_points_a[1][0][1]) > abs(end_points_a[0][0][0] - end_points_a[3][0][0]) and abs(end_points_b[0][0][1] - end_points_b[1][0][1]) > abs(end_points_b[0][0][0] - end_points_b[3][0][0])):
					x = []
					y = []
					x.append(end_points_a[0][0][0])
					x.append(end_points_a[3][0][0])
					x.append(end_points_b[0][0][0])
					x.append(end_points_b[3][0][0])
					x.sort()
					x_left = x[1]
					x_right = x[2]
					y.append(end_points_a[0][0][1])
					y.append(end_points_a[1][0][1])
					y.sort()
					y_bottom = y[1]
					#filename = 'save.jpg'
					plate = image_copy[y_bottom-40:y_bottom + 10, x_left:x_right]
					self.plate_pub.publish(self.bridge.cv2_to_imgmsg(plate, "bgr8"))
					print("published")
					#cv2.imwrite(filename, plate)
		cv2.waitKey(1)
		cv2.destroyAllWindows()

def main(args):
	rospy.init_node("license_extraction", anonymous=True)
	ic = license_extraction()
	rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
   
