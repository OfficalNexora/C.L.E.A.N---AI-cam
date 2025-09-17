import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -----------------------
# -----------------------
# Training Settings
# -----------------------
# -----------------------
TRAIN_DIR = "dataset/images"   # path 
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# MobileNetV2 trainable settings
FREEZE_LAYERS = 60        # how many layers to freeze from the bottom
DENSE_UNITS = 64          # size of dense layer
DROPOUT_RATE = 0.4        # dropout to prevent overfitting

# Learning & optimization
LEARNING_RATE = 1e-4
LR_PATIENCE = 3            # epochs before reducing learning rate
EARLYSTOP_PATIENCE = 5     # epochs before early stop

# Augmentation settings
ROTATION_RANGE = 30
ZOOM_RANGE = 0.25
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
SHEAR_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
BRIGHTNESS_RANGE = [0.7, 1.3]

# Probabilities for custom augmentations
PROBABILITY_BRIGHTNESS = 0.5
PROBABILITY_CONTRAST = 0.5
PROBABILITY_NOISE = 0.3
PROBABILITY_BLUR = 0.3
PROBABILITY_BACKGROUND = 0.3
PROBABILITY_HUE = 0.2

USE_CUSTOM_PREPROCESSING = True

# -----------------------
# Custom augmentation functions
# -----------------------
def add_noise(img):
    row, col, ch = img.shape
    sigma = 0.02
    gauss = np.random.normal(0, sigma, (row, col, ch))
    noisy = img + gauss
    return np.clip(noisy, 0, 1)

def add_blur(img):
    if np.random.rand() < 0.5:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img

def add_brightness(img):
    factor = np.random.uniform(0.7, 1.3)
    return np.clip(img * factor, 0, 1)

def add_contrast(img):
    factor = np.random.uniform(0.8, 1.2)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 1)

def synthetic_background(img):
    if np.random.rand() < 0.3:
        bg_color = np.random.uniform(0.7, 1.0)
        bg_patch = np.ones_like(img) * bg_color
        alpha = np.random.uniform(0.1, 0.3)
        img = img * (1 - alpha) + bg_patch * alpha
        return np.clip(img, 0, 1)
    return img

def add_hue(img):
    factor = np.random.uniform(-0.1, 0.1)
    img_hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
    img_hsv[:,:,0] = (img_hsv[:,:,0] + factor) % 1.0
    img_rgb = cv2.cvtColor((img_hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
    return img_rgb

def preprocessing_function(img):
    img = img / 255.0 # normalize to [0,1]
    if USE_CUSTOM_PREPROCESSING:
        # Brightness: 
        if np.random.rand() < PROBABILITY_BRIGHTNESS:
            img = add_brightness(img)
            
        # Contrast: 
        if np.random.rand() < PROBABILITY_CONTRAST:
            img = add_contrast(img)
        
        # Noise: 
        if np.random.rand() < PROBABILITY_NOISE:
            img = add_noise(img)
        
        # Blur: 
        if np.random.rand() < PROBABILITY_BLUR:
            img = add_blur(img)
        
        # Synthetic background: 
        if np.random.rand() < PROBABILITY_BACKGROUND:
            img = synthetic_background(img)

        # Hue shift: 
        if np.random.rand() < PROBABILITY_HUE:
            img = add_hue(img)
    return img

# -----------------------
# Data Augmentation
# -----------------------
train_datagen = ImageDataGenerator(
    rotation_range=ROTATION_RANGE,
    zoom_range=ZOOM_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    vertical_flip=VERTICAL_FLIP,
    shear_range=SHEAR_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    brightness_range=BRIGHTNESS_RANGE,
    preprocessing_function=preprocessing_function if USE_CUSTOM_PREPROCESSING else None,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function if USE_CUSTOM_PREPROCESSING else None,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -----------------------
# Load existing model OR create a new one
# -----------------------
try:
    model = load_model("Clean.keras")
    print("Loaded existing model.")
except (OSError, FileNotFoundError):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = True

    for layer in base_model.layers[:-FREEZE_LAYERS]:
        if not isinstance(layer, layers.BatchNormalization):
             layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(DENSE_UNITS, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    print("Created new model from scratch.")

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# -----------------------
# Callbacks
# -----------------------
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=LR_PATIENCE, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True)

# -----------------------
# Class Weighting
# -----------------------
class_weights = {i: 1.0 for i in range(train_generator.num_classes)}
print("Class indices:", train_generator.class_indices)

# -----------------------
# Train Model
# -----------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stop],
    class_weight=class_weights,
    verbose=1
)

# -----------------------
# Save Model
# -----------------------
model.save("Clean.keras")
print("Model saved.")

# -----------------------
# Plot Training Curves
# -----------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.show()
