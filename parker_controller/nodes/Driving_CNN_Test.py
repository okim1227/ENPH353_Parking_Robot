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

import ipywidgets as ipywidgets
from ipywidgets import interact

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend

from gazebo_msgs.msg import ModelStates

# Instantiate CvBridge
br = CvBridge()

class Driving_CNN_Test:

  def __init__(self):

    # self.car_maual_driver_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.CNN_callback)
    self.car_maual_driver_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.car_manual_driver_callback)
    self.camera_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback, queue_size=10)
    self.speed  = []
    self.twist_data = []
    self.camera_data = []
    time.sleep(1)

  def car_manual_driver_callback(self, msg):
    self.speed = msg

  def camera_callback(self, msg):
    print("Received an image!")
    #publish to the camera node to take a picture and save everytime we get a message
    self.camera_data.append(msg)
    self.twist_data.append(self.speed)
    # print(self.camera_data)
    # print(self.twist_data)

  def convert_to_one_hot(twist_command):
    array = ['Straight', 'Turn_Left']
    index = array.index(twist_command)
    Y= np.eye(2)[index]
    return Y
  
  def displayImage(index):
    plt.imshow(X_dataset[index])
    caption = ("y = " + str(Y_dataset[index]))#str(np.squeeze(Y_dataset_orig[:, index])))
    plt.text(0.5, 0.5, caption, 
             color='orange', fontsize = 20,
             horizontalalignment='left', verticalalignment='top')

  def reset_weights(model):
      session = backend.get_session()
      for layer in model.layers: 
          if hasattr(layer, 'kernel_initializer'):
              layer.kernel.initializer.run(session=session)

  def predict(image):
    datas = self.twist_data
    imgset = self.camera_data

    np.random.shuffle(imgset)

    X_dataset_orig = np.array([data[0] for data in imgset[:]])
    X_dataset = X_dataset_orig/255
    Y_dataset_orig = np.array([[data[1]] for data in imgset])
    Y_dataset = np.array([convert_to_one_hot(letter) for letter in Y_dataset_orig])

    VALIDATION_SPLIT = 0.2

    print("Total examples: {:d}\nTraining examples: {:d}\nTest examples: {:d}".
          format(X_dataset.shape[0],
                 math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)),
                 math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))
    
    interact(displayImage, 
           index=ipywidgets.IntSlider(min=0, max=X_dataset_orig.shape[0],
                                      step=1, value=10))
    displayImage(35)

    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                 input_shape=(125, 105, 3)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(125, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(105, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dropout(0.5))
    conv_model.add(layers.Dense(512, activation='relu'))
    conv_model.add(layers.Dense(36, activation='softmax'))

    conv_model.summary()

    LEARNING_RATE = 1e-4
    conv_model.compile(loss='categorical_crossentropy',
                       optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                       metrics=['acc'])

    reset_weights(conv_model)

    history_conv = conv_model.fit(X_dataset, Y_dataset, 
                                  validation_split=VALIDATION_SPLIT, 
                                  epochs=80, 
                                  batch_size=16)

    conv_model.save("driving_CNN_model.h5")


rospy.init_node("Driving_CNN_Test", anonymous=True)

def main(args):
    ic = Driving_CNN_Test()
    time.sleep(1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)