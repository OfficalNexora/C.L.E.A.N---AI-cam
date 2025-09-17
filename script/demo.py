import keras

# Load your model
model = keras.models.load_model("Clean.keras")

# See the architecture
model.summary()

# If you want the raw weights
weights = model.get_weights()
print(len(weights))  # number of layers' weight sets