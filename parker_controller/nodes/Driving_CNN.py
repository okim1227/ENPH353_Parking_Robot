import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import numpy as np


sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)

driving_NN = models.load_model("driving_CNN_model.h5")

def predict(image):
	global sess1
	global graph1
	with graph1.as_default():
		set_session(sess1)
		NN_prediction = driving_NN.predict(image)[0]
	return NN_prediction

blank_image = np.array([np.zeros((125, 105, 3))])

print(predict(blank_image))