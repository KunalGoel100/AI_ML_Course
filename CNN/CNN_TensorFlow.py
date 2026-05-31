import tensorflow as tf
import numpy as np

# Load dataset (MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_test[1].reshape(1,28,28,1))

####################################################
## Visualise the image
import matplotlib.pyplot as plt

plt.imshow(x_test[3], cmap='gray')
plt.title(f"Label: {y_test[3]}")
plt.axis('off')   # removes axis numbers
plt.show()

###################################################
# Keep only digits 1–9 (remove 0 if needed)

# Normalize (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension (for CNN)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Model
model = tf.keras.Sequential([

    # Convolution layer
    tf.keras.layers.Conv2D(16, (3,3), activation='elu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    # # Another conv layer
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),

    # Flatten for dense layer
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(64, activation='elu'),

    # Output layer (digits 1–9 → 9 classes)
    tf.keras.layers.Dense(10, activation='softmax')  # MNIST still outputs 0–9
])

# Compile
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

# Test prediction
pred = model.predict(x_test[3].reshape(1,28,28,1))
print("Prediction:", np.argmax(pred))