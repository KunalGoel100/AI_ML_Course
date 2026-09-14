import tensorflow as tf
import numpy as np

import cv2

# # Read image in grayscale
# x_img = cv2.imread(
#     r"C:\KunalGoel\AI_ML_Course\CNN\HandWrittenNumber_CNN\Data\Two_2.jpg",
#     cv2.IMREAD_GRAYSCALE
# )
# # Resize to 28x28
# x_img = cv2.resize(x_img, (28, 28))
#
# # Normalize pixel values
# x_img = x_img / 255.0
#
# # Reshape for CNN
# x_img = x_img.reshape(1, 28, 28, 1)
# x_img = 1-x_img
# #
# ####################################################
# # # Visualise the image
# import matplotlib.pyplot as plt
# #
# plt.imshow(x_img[0], cmap='gray')
# # plt.title(f"Label: {y_test[3]}")
# plt.axis('off')   # removes axis numbers
# plt.show()

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
###############################################################
img = cv2.imread(r"C:\KunalGoel\AI_ML_Course\CNN\HandWrittenNumber_CNN\Data\Exp_4.jpg", 0)

_, thresh = cv2.threshold(
    img,
    127,
    255,
    cv2.THRESH_BINARY_INV
)
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
boxes = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w > 8 and h > 8:
        boxes.append((x,y,w,h))
boxes = sorted(boxes,key=lambda b:b[0])
# print(boxes)
expression = ""
for x,y,w,h in boxes:
    roi = thresh[y:y+h, x:x+w]
    # Keep aspect ratio
    scale = 20.0 / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    roi = cv2.resize(roi,(new_w,new_h))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h,x_offset: x_offset + new_w] = roi
    roi = canvas.astype("float32") / 255.0
    roi = roi.reshape(1, 28, 28, 1)
    # # Visualise the image
    import matplotlib.pyplot as plt

    plt.imshow(roi[0], cmap='gray')
    # plt.title(f"Label: {y_test[3]}")
    plt.axis('off')  # removes axis numbers
    plt.show()
###############################################################
    pred = model.predict(roi)
    # print(pred)
    print("Prediction:", np.argmax(pred))
    print("Confidence:", np.max(pred))
    Choice = input("Is the prediction Correct: Y , N")
    if Choice == "Y" or Choice == "y":
        # model.fit(x_img,np.array([np.argmax(pred)]), epochs=1)
        # model.save_weights("SavedWeights.weights.h5")
        # print("Weights Updated")
        expression += str(np.argmax(pred))
    elif Choice == "N" or Choice == "n":
        correct = int(input("Enter correct number: "))
        model.fit(roi, np.array([correct]), epochs=3)
        model.save_weights("SavedWeights.weights.h5")
        print("Weights Updated")
        expression += str(correct)
    else:
        pass
print(f"Expression: {expression}")

