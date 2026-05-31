import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sympy.codegen.rewriting import optimize

#
# plt.imshow(x_test[5], cmap='gray')
# plt.title(f"Label: {y_test[5]}")
# plt.axis('off')   # removes axis numbers
# plt.show()

IMG_SIZE = 4
BATCH_SIZE = 10

# Create random images
images = tf.random.uniform((100, IMG_SIZE, IMG_SIZE, 3))

# Create fake masks (simple rule)
masks = tf.cast(images[:,:,:,0:1] > 0.5, tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((images, masks))
dataset = dataset.batch(BATCH_SIZE)

# print(images[0])
# plt.imshow(masks[0])
# plt.title("Image")
# plt.show()
#
# Model
model = tf.keras.Sequential([
#
    # Convolution layer
    tf.keras.layers.Conv2D(64, (3,3),input_shape=(None,None,3),padding='same',activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3,3),padding='same',activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(16, (3,3),padding='same',activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),

    # tf.keras.layers.Conv2DTranspose(16,(3,3),strides=2,padding='same'),
    # tf.keras.layers.Conv2DTranspose(32,(3,3),strides=2,padding='same'),
    # tf.keras.layers.Conv2DTranspose(64,(3,3),strides=2,padding='same'),

    # Output
    tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

])
#
# # Compile
# model.layers[0].trainable = False
# model.layers[1].trainable = False
# model.summary()
# model.compile(
#     optimizer='sgd',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model.fit(dataset, epochs=100)
# model.save_weights("base_model.weights.h5")
# #
# Train2
# Create random images
images2 = tf.random.uniform((10, 8, 8, 3))

# Create fake masks (simple rule)
masks2 = tf.cast(images2[:,:,:,0:1] > 0.5, tf.float32)

dataset2 = tf.data.Dataset.from_tensor_slices((images2, masks2))
dataset2 = dataset2.batch(BATCH_SIZE)

# # Compile
model.layers[0].trainable = False
model.layers[1].trainable = False
model.summary()
model.load_weights("base_model.weights.h5")
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(dataset2, epochs=100)
