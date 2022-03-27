#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('parker_controller')
import sys
import rospy
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

class CNN_Test:

  def __init__(self):

    self.car_maual_driver_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.CNN_callback)

  def CNN_callback(self, data):
    print(data)

def main(args):
    ic = CNN_Test()

if __name__ == '__main__':
    main(sys.argv)
     