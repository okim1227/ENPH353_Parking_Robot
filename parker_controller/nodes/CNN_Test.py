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

from gazebo_msgs.msg import ModelStates

# Instantiate CvBridge
br = CvBridge()

class CNN_Test:

  def __init__(self):

    self.car_maual_driver_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.CNN_callback)
    self.camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback)

    time.sleep(1)

  def CNN_callback(self, msg):
    print(msg)

  def camera_callback(self, msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('camera_image.jpeg', cv2_img)

    self.image = self.bridge.imgmsg_to_cv2(msg)

  def start(self):
    rospy.loginfo("Timing images")
    while not rospy.is_shutdown():
        rospy.loginfo('publishing image')
        if self.image is not None:
            self.pub.publish(br.cv2_to_imgmsg(self.image))
        self.loop_rate.sleep()


rospy.init_node("CNN_Test", anonymous=True)

def main(args):
    ic = CNN_Test()

if __name__ == '__main__':
    main(sys.argv)