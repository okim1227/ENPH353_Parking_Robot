#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import time
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
		gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
		_, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
		crop_img = binary[450:600, 0:350]
		plt.imshow(crop_img, cmap='gray', vmin=0, vmax=255)
		plt.show()

def main(args):
	rospy.init_node("license_extraction", anonymous=True)
	ic = license_extraction()
	rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
   
