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

class driver:

  def __init__(self):

    #self.bridge = CvBridge()

    #self.pedestrian_sub = rospy.Subscriber("/R1/camera1/image_raw",Image,self.callback)
    #self.truck_sub = rospy.Subscriber("/R1/camera1/image_raw",Image,self.callback)
    
    self.camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback)

    self.cmdVel_pub = rospy.Publisher("/R1/cmd_vel",Twist)
    move = Twist() 
    self.license_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    #self.clock_sub = rospy.Subscriber("/clock",String,self.clock_callback)

    time.sleep(1)

    self.license_pub.publish("NO_NAME,multi12,0,XR58")
    
    time.sleep(1)

    move.linear.x = 0.40
    self.cmdVel_pub.publish(move)
    time.sleep(0.52)
    move.linear.x = 0.0
    self.cmdVel_pub.publish(move)

    move.linear.x = 0.5
    move.angular.z = 4.5
    self.cmdVel_pub.publish(move)
    time.sleep(0.4)
    move.linear.x = 0.4
    move.angular.z = 0.0
    self.cmdVel_pub.publish(move)
    time.sleep(1)
    move.linear.x = 0.0
    self.cmdVel_pub.publish(move)
    self.license_pub.publish("NO_NAME,multi12,-1,XR58")


  #def clock_callback(self,data):
    #print(data)
  def camera_callback(self,data):
    print(data)


rospy.init_node("driver", anonymous=True)

def main(args):
  ic = driver()

if __name__ == '__main__':
  main(sys.argv)
   