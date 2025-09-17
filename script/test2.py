import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Load Sequential model
seq_model = tf.keras.models.load_model("Clean.keras")
base_model = seq_model.layers[0]  # MobileNetV2 base

# Load image
img_path = "dataset/images/paper/paper116.png"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# Get convolutional layers
conv_layers = [layer for layer in base_model.layers 
               if 'conv' in layer.name or 'depthwise' in layer.name]
print(f"Using layers for activation visualization: {[l.name for l in conv_layers[:5]]}")

# Visualize activations
def visualize_activations(base_model, img_tensor, conv_layers, max_filters=8, delay=2):
    for layer in conv_layers[:5]:  # first 5 layers for speed
        intermediate_model = Model(inputs=base_model.input, outputs=layer.output)
        activations = intermediate_model.predict(img_tensor)

        num_filters = min(activations.shape[-1], max_filters)
        n_cols = min(num_filters, 8)
        n_rows = (num_filters // n_cols) + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        axes = axes.flatten()

        for i in range(num_filters):
            axes[i].imshow(activations[0, :, :, i], cmap='viridis')
            axes[i].axis('off')

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Activations of {layer.name}", fontsize=14)
        plt.show(block=True)
        plt.pause(delay)
        #plt.close()

# Run visualization
visualize_activations(base_model, x, conv_layers)