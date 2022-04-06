#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Instantiate CvBridge
br = CvBridge()
cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')

class Plate_Imaging:

	def __init__(self):
		frame = "/../camera_captures/plate2_1.png"

		rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		print(rgb.shape)
		plt.imshow(rgb)


rospy.init_node("Plate_Imaging", anonymous=True)


def main(args):
	ic = Plate_Imaging()
	time.sleep(1)

	rospy.spin()

if __name__ == '__main__':
    main(sys.argv)

