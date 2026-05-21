import os

from sympy.physics.units import action

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

# data = [[0,0,1],[0,1,0],[1,0,0],[1,1,0]]
# data_x = np.array(  [1,2,3,4,5,6,7,8,9,10],dtype=float)
# data_x = np.array([x%2 for x in data_x])
data_x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
print(data_x)
data_y = np.array([0,1,1,1],dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[2],activation='sigmoid')])
model.compile(optimizer='sgd',
              loss='binary_crossentropy')
model.fit(data_x,data_y,epochs=100)

print(model.get_weights())
print(model.predict(np.array([[0,1]],dtype=float)))

