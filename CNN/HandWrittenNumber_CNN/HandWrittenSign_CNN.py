import tensorflow as tf
import numpy as np
import os
import cv2
from sympy.plotting.textplot import rescale

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
    tf.keras.layers.Dense(5, activation='softmax')  # MNIST still outputs 0–9
])
#
# # Compile
model.load_weights("SavedWeights_Sign.weights.h5")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#
# # Train
# model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
# model.save_weights("SavedWeights_Number.weights.h5")
# Test prediction
###############################################################
classes = {
    "addition": 0,
    "division": 1,
    "multiplication": 2,
    "subtraction": 3
}
# base_folder = r"C:\KunalGoel\AI_ML_Course\CNN\HandWrittenNumber_CNN\HMO5_Final"
#
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     base_folder,
#     labels="inferred",
#     label_mode="int",
#     image_size=(28, 28),
#     batch_size=32,
#     shuffle=True,
#     color_mode="grayscale"
# )
# print(train_ds.class_names)
# class_names = train_ds.class_names
# train_ds = train_ds.map(
#     lambda x, y: ((255.0 - x) / 255.0, y)
# )
#
# model.fit(
#     train_ds,
#     epochs=20
# )
# model.save_weights(r"SavedWeights_Sign.weights.h5")
###############################################################
img = cv2.imread(r"C:\KunalGoel\AI_ML_Course\CNN\HandWrittenNumber_CNN\Data2\Exp_3.jpg", 0)

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
    # Keep aspect ratioa
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
    pred = model.predict(roi)
    print(pred)
    print(np.argmax(pred))
    ans = [key for key,val in classes.items() if val == np.argmax(pred)]
    print(ans)
