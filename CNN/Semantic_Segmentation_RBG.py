import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

####################################################
## 1. Paths and Data Loading
image_folder = r"C:\KunalGoel\AI_ML_Course\CNN\Dataset\FootBallMatch_Sementic\images"
mask_folder = r"C:\KunalGoel\AI_ML_Course\CNN\Dataset\FootBallMatch_Sementic\Masks"

image_dataset = tf.keras.utils.image_dataset_from_directory(
    image_folder, label_mode=None, image_size=[256, 256], batch_size=10, shuffle=False
)
mask_dataset = tf.keras.utils.image_dataset_from_directory(
    mask_folder, label_mode=None, image_size=[256, 256], batch_size=10, shuffle=False
)
# mask = [m for m in mask_dataset]
# image = [i for i in image_dataset]
# plt.subplot(1,2,1)
# plt.imshow(mask[0][2,:,:,2])
# plt.subplot(1,2,2)
# plt.imshow(np.array(image[0][2,:,:,:], dtype=np.uint8))
# plt.show()


####################################################
## 2. Define the 11 Class Colors
# 🔴 MUST BE CHANGED: Replace these with your exact 11 RGB values.
# The index order determines the integer ID (0 to 10) assigned to that color.
CLASS_COLORS = np.array([
    [255,0,29],  # Class 0: Goal pole
    [255,160,1],  # Class 1: P1 red
    [254,233,3],  # Class 2: P2 black
    [27,71,151], #class 3: add
    [111,48,253], #class4: back
    [201,19,223], #class5: ball
    [137,126,126], #class6: ground
    [238,164.5,171], #class7: refree
    [255,159,0], #class8: goalkeeper1
    [109,112.2,151], #class9: back pole
    [255,235,0], #class10: goalkeeper2
], dtype = np.int32)

def clean_pipeline(image, mask):
    # Normalize input images to 0.0 - 1.0 range
    image = image / 255.0

    # Broadcast mathematical comparison to map RGB colors to 0-10 class indices
    colors = tf.reshape(CLASS_COLORS, (1, 1, 1, len(CLASS_COLORS), 3))
    colors = tf.cast(colors, tf.float32)
    mask_expanded = tf.expand_dims(tf.cast(mask, tf.float32), axis=-2)
    distances = tf.reduce_sum(tf.abs(mask_expanded - colors), axis=-1)

    # Find closest color match index per pixel
    one_channel_mask = tf.argmin(distances, axis=-1)
    one_channel_mask = tf.expand_dims(one_channel_mask, axis=-1)  # Shape: (Batch, 256, 256, 1)

    return image, one_channel_mask


# Build optimization pipeline
dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
dataset = dataset.map(clean_pipeline).prefetch(tf.data.AUTOTUNE)

####################################################
## 3. Model Definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(None, None, 3), padding='same', activation='elu'),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='elu'),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='elu'),
    tf.keras.layers.Conv2D(11, (1, 1), activation='softmax')  # 11 channel output matching 11 classes
])

# Compile with proper multi-class settings
# model.load_weights("base_rgb_test.weights.h5")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Correct loss for 1-channel integer indices
    metrics=['accuracy']
)

# Train the model
model.fit(dataset, epochs=20)
# model.save_weights("base_rgb_test.weights.h5")
#
# ###################################
## testing
image = [i for i in image_dataset]
image = image[0]
image = image/255.0
# print(image)
pred = model.predict(np.array(image[:1]))
# print(pred.shape)
pred_mask = np.argmax(pred[0], axis = -1)
# print(pred_mask)
colour_mask = CLASS_COLORS[pred_mask]
print(colour_mask.shape)
plt.imshow(np.array(colour_mask, dtype=np.uint8))
plt.show()
