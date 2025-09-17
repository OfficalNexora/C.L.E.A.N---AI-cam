import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tensorflow.keras.preprocessing import image
import umap  # faster and more animation-friendly than t-SNE

# ------------------------
# Load model
# ------------------------
seq_model = tf.keras.models.load_model("Clean.keras")
base_model = seq_model.layers[0]

# ------------------------
# Load dataset
# ------------------------
dataset_dir = "dataset/images"
classes = ["paper", "plastic"]

images = []
labels = []
filenames = []

for idx, cls in enumerate(classes):
    cls_dir = os.path.join(dataset_dir, cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(cls_dir, fname)
            img = image.load_img(img_path, target_size=(224,224))
            x = image.img_to_array(img) / 255.0
            images.append(x)
            labels.append(idx)
            filenames.append(img_path)

images = np.array(images)
labels = np.array(labels)
print(f"Loaded {len(images)} images.")

# ------------------------
# Extract features
# ------------------------
features = base_model.predict(images)
features_flat = features.reshape(features.shape[0], -1)
print(f"Flattened features: {features_flat.shape}")

# ------------------------
# UMAP reduction
# ------------------------
reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
embedding = reducer.fit_transform(features_flat)
print(f"UMAP embedding shape: {embedding.shape}")

# ------------------------
# Interactive plot with thumbnails
# ------------------------
def imscatter(x, y, images, ax=None, zoom=0.15):
    if ax is None:
        ax = plt.gca()
    artists = []
    for x0, y0, img in zip(x, y, images):
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
        artists.append(ab)
    return artists

# Prepare thumbnail images
thumbs = [image.img_to_array(image.load_img(f, target_size=(32,32)))/255.0 for f in filenames]

fig, ax = plt.subplots(figsize=(10,8))
colors = ['red', 'blue']

for cls_idx, cls_name in enumerate(classes):
    idxs = labels==cls_idx
    ax.scatter(embedding[idxs,0], embedding[idxs,1], color=colors[cls_idx], label=cls_name, alpha=0.3)

imscatter(embedding[:,0], embedding[:,1], thumbs, ax=ax, zoom=0.15)
ax.legend()
ax.set_title("UMAP: AI Mind Map of Paper vs Plastic")
plt.show()