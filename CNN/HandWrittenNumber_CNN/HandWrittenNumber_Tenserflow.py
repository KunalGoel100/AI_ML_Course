import tensorflow as tf
import numpy as np

import cv2

# Read image in grayscale
x_img = cv2.imread(
    r"C:\KunalGoel\AI_ML_Course\CNN\HandWrittenNumber_CNN\Data\Eight_2.jpg",
    cv2.IMREAD_GRAYSCALE
)
# Resize to 28x28
x_img = cv2.resize(x_img, (28, 28))

# Normalize pixel values
x_img = x_img / 255.0

# Reshape for CNN
x_img = x_img.reshape(1, 28, 28, 1)
x_img = 1-x_img
#
# ####################################################
# # Visualise the image
import matplotlib.pyplot as plt
#
plt.imshow(x_img[0], cmap='gray')
# plt.title(f"Label: {y_test[3]}")
plt.axis('off')   # removes axis numbers
plt.show()

###################################################
# # Load dataset (MNIST)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# # Normalize (0–255 → 0–1)
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#
# # Add channel dimension (for CNN)
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
#
# # plt.imshow(x_train[1], cmap='gray')
# # plt.title(f"Label: {y_test[3]}")
# plt.axis('off')   # removes axis numbers
# plt.show()

# Model
model = tf.keras.Sequential([

    # Convolution layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    # Another conv layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # Flatten for dense layer
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(64, activation='relu'),

    # Output layer (digits 1–9 → 9 classes)
    tf.keras.layers.Dense(10, activation='softmax')  # MNIST still outputs 0–9
])

# Compile
model.load_weights("SavedWeights.weights.h5")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
# model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
# model.save_weights("SavedWeights.weights.h5")
# Test prediction

pred = model.predict(x_img)
print(pred)
print("Prediction:", np.argmax(pred))
print("Confidence:", np.max(pred))
Choice = input("Is the prediction Correct: Y , N")
if Choice == "Y" or Choice == "y":
    model.fit(x_img,np.array([np.argmax(pred)]), epochs=1)
    model.save_weights("SavedWeights.weights.h5")
else:
    correct = int(input("Enter correct number: "))
    model.fit(x_img, np.array([correct]), epochs=1)
    model.save_weights("SavedWeights.weights.h5")