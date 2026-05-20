import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np

# data = [[1,2],[2,4],[3,6],[4,8],[5,10]]
data_x = np.array([1,2,3,4,5,6,7,8,9],dtype=float)
data_y = np.array([2,4,6,8,10,12,14,16,18],dtype=float)


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',
              loss='mean_squared_error')
model.fit(data_x,data_y,epochs=500)
print(model.predict(np.array([[1]],dtype=int)))
print(model.get_weights())

