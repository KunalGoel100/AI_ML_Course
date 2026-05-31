import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sympy.codegen.rewriting import optimize

def create_circle_dataset(num_samples, img_size):
    images = []
    masks = []

    for _ in range(num_samples):
        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        mask = np.zeros((img_size, img_size, 1), dtype=np.float32)

        # Random circle parameters
        center_x = np.random.randint(img_size//4, 3*img_size//4)
        center_y = np.random.randint(img_size//4, 3*img_size//4)
        radius = np.random.randint(img_size//8, img_size//4)

        for y in range(img_size):
            for x in range(img_size):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    img[y, x] = [1, 1, 1]   # white circle
                    mask[y, x] = 1

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


def create_square_dataset(num_samples, img_size):
    images = []
    masks = []

    for _ in range(num_samples):
        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        mask = np.zeros((img_size, img_size, 1), dtype=np.float32)

        # Random square parameters
        center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
        center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
        half_size = np.random.randint(img_size // 8, img_size // 4)
        half_size2 = half_size/1.5

        for y in range(img_size):
            for x in range(img_size):
                if (abs(x - center_x) <= half_size and
                        abs(y - center_y) <= half_size):
                    img[y, x] = [1, 1, 1]  # white square
                    mask[y, x] = 1
        for y in range(img_size):
            for x in range(img_size):
                if (abs(x - center_x) <= half_size2 and
                        abs(y - center_y) <= half_size2):
                    img[y, x] = [0, 0, 0]  # white square
                    mask[y, x] = 0


        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


#
# Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3),input_shape=(None,None,3),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(32, (3,3),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(16, (3,3),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
])
#
# images, masks = create_circle_dataset(100,32)
images, masks = create_square_dataset(100,64)
# plt.subplot(1,2,1)
# plt.imshow(images[0,:,:,0])
# plt.subplot(1,2,2)
# plt.imshow(images[0,:,:,1])
# plt.show()
# plt.imshow(masks[0])
# plt.show()
# print(images.shape)

model.load_weights("base_Weights_Shape.weights.h5")
model.layers[0].trainable = False
model.layers[1].trainable = False
# model.layers[2].trainable = False

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()
model.fit(images,masks, epochs=10)
# model.save_weights("base_Weights_Shape.weights.h5")

pred = model.predict(images[:1])
output = tf.cast(pred[0,:,:] > 0.5, tf.float32)

plt.subplot(1,2,1)
plt.imshow(images[0,:,:,:])
plt.title("Input")
plt.subplot(1,2,2)
plt.title("Output")
plt.imshow(output)
plt.show()
# #with
# # # Test prediction
# # pred = model.predict(x_test[5].reshape(1,28,28,1))
# # print