#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

class driver:

  def __init__(self):

    self.bridge = CvBridge()

    #self.pedestrian_sub = rospy.Subscriber("/rrbot/camera1/image_raw",Image,self.callback)
    #self.truck_sub = rospy.Subscriber("/rrbot/camera1/image_raw",Image,self.callback)
    
    #self.camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback)
    self.clock_sub = rospy.Subscriber("/clock",String,self.clock_callback)

    #self.driver_pub = rospy.Publisher("/driver",Twist)
    #self.cmdVel_pub = rospy.Publisher("/R1/cmd_vel",Twist)
    self.license_pub = rospy.Publisher("/license_plate", String)

    self.license_pub.publish(str('TeamRed,multi12, -1, XR58'))


  #def clock_callback(self,data):
    #print data


def main(args):
  ic = driver()
  rospy.init_node("driver", anonymous=True)


if __name__ == '__main__':
  main(sys.argv)
   
