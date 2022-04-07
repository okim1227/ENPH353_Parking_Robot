#!/usr/bin/env python
from __future__ import print_function

import roslib
import numpy as np
import math
#roslib.load_manifest('parker_controller')
import sys
import rospy
import cv2
import time

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

from gazebo_msgs.msg import ModelStates

import os
directory = r'/home/fizzer/ros_ws/src/parker_controller/nodes'
os.chdir(directory)

class Driving_CNN_Test:

  def __init__(self):
    # self.car_maual_driver_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.CNN_callback)
    self.car_maual_driver_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.car_manual_driver_callback)
    self.camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback, queue_size=100)
    self.bridge = CvBridge()
    self.speed  = []
    self.twist_data = []
    self.capture_index = 0
    time.sleep(1)

  def car_manual_driver_callback(self, msg):
    self.speed_linear = msg

  def camera_callback(self, msg):
    #print("Received an image!")

    # Append the current twist command when camera callback is called
    self.twist_data.append(self.speed)
    print(self.twist_data)

    # output_twist_file = open("Twist_Commands.txt", "w")
    # output_twist_file.write(self.twist_data)
    # output_twist_file.close()

    # Save the captured image from camera subscriber to the Driving Views folder
    # Name of capture is Capture_{linear_x command}_{angular_z command}
    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite(os.path.join(directory + "/Driving_Data/", "Capture_{}_{}.png".format(self.speed.linear.x, self.speed.angular.z)), cv_image)
    
    self.capture_index = self.capture_index + 1
    print(self.capture_index)

rospy.init_node("Driving_CNN_Test", anonymous=True)

def main(args):
    ic = Driving_CNN_Test()
    time.sleep(1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)