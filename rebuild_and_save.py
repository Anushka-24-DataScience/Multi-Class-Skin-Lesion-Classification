import os
import tensorflow as tf

print(f"TF version: {tf.__version__}")

# Rebuild exact same architecture locally
base = tf.keras.applications.EfficientNetB4(
    input_shape=(380, 380, 3),
    weights=None,
    include_top=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.models.Model(inputs=base.input, outputs=output)
print("✅ Architecture rebuilt")

# Load Colab trained weights
model.load_weights("artifacts/training/trained_weights.weights.h5")
print("✅ Weights loaded successfully")

# Save in local Keras 2.12 compatible H5 format
model.save("artifacts/training/trained_model_local.h5")
print("✅ Model saved as trained_model_local.h5")

# Verify
loaded = tf.keras.models.load_model(
    "artifacts/training/trained_model_local.h5",
    compile=False
)
print(f"✅ Verification successful - Input: {loaded.input_shape}, Output: {loaded.output_shape}")