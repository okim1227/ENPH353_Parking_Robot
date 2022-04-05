#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt 

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

class license_extraction:

	def __init__(self):
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
		#ret, thresh = cv2.threshold(crop_img, 70, 255, cv2.THRESH_BINARY)
		image, contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
		print(contours)
		#cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
		cv2.imshow('None approximation', image_copy)
		cv2.waitKey(0)
		cv2.imwrite('contours_none_image1.jpg', image_copy)
		cv2.destroyAllWindows()

def main(args):
	rospy.init_node("license_extraction", anonymous=True)
	ic = license_extraction()
	rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
   
