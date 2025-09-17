import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# ------------------------
# Load model
# ------------------------
seq_model = tf.keras.models.load_model("Clean.keras")
base_model = seq_model.layers[0]  # MobileNetV2
top_layers = seq_model.layers[1:]  # rest of Sequential

# ------------------------
# Load image
# ------------------------
img_path = "dataset/images/paper/paper110.png"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# ------------------------
# Find all Conv2D layers
# ------------------------
conv_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
print(f"Using layers for Grad-CAM++: {[l.name for l in conv_layers]}")

# ------------------------
# Grad-CAM++ computation
# ------------------------
def compute_gradcam_plus_plus(model, img_tensor, conv_layer):
    conv_output = conv_layer.output
    x_top = base_model.output
    for layer in top_layers:
        x_top = layer(x_top)
    grad_model = Model(inputs=base_model.input, outputs=[conv_output, x_top])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Grad-CAM++
    grads = tape.gradient(loss, conv_outputs)[0]  # remove batch dim
    first_derivative = grads
    second_derivative = tf.square(grads)
    third_derivative = tf.pow(grads, 3)

    global_sum = tf.reduce_sum(conv_outputs[0], axis=(0,1))
    alpha_num = second_derivative
    alpha_denom = 2*second_derivative + global_sum*third_derivative + 1e-8
    alpha = alpha_num / alpha_denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(first_derivative), axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam.numpy(), (224,224))
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    return cam

# Compute and fuse all CAMs
cams = [compute_gradcam_plus_plus(base_model, x, l) for l in conv_layers]
# Weighted fusion: deeper layers get higher weight
weights = np.linspace(0.5, 1.5, len(cams))
fused_cam = np.zeros_like(cams[0])
for w, cam in zip(weights, cams):
    fused_cam += w * cam
fused_cam /= np.sum(weights)

# ------------------------
# Overlay
# ------------------------
heatmap = cv2.applyColorMap(np.uint8(255*fused_cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(
    cv2.cvtColor(np.uint8(x[0]*255), cv2.COLOR_RGB2BGR),
    0.5,
    heatmap,
    0.5,
    0
)

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted class index: {np.argmax(seq_model.predict(x)[0])}")
plt.axis("off")
plt.show()