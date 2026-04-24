# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import tensorflow as tf
# import numpy as np

# print("TF version:", tf.__version__)

# # Try loading with custom object scope
# print("Loading model...")
# with tf.keras.utils.custom_object_scope({}):
#     model = tf.keras.models.load_model(
#         r"HF_space_deployment/artifacts/training/trained_model_local.h5",
#         compile=False
#     )
# print("Model loaded!")
# print("Input shape:", model.input_shape)
# print("Output shape:", model.output_shape)

# # Rebuild model by cloning to strip any unpicklable objects
# print("Cloning model...")
# new_model = tf.keras.models.clone_model(model)
# new_model.set_weights(model.get_weights())
# print("Model cloned!")

# # Save as new h5
# print("Saving...")
# new_model.save(
#     r"HF_space_deployment/artifacts/training/trained_model_local.h5",
#     save_format="h5"
# )
# print("Done! Model re-saved successfully.")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

print("TF:", tf.__version__)

model = tf.keras.models.load_model(
    "artifacts/training/trained_model.keras",
    compile=False
)
print("✅ Loaded")

# Save as H5 — version independent
model.save(
    "HF_space_deployment/artifacts/training/trained_model_hf.h5",
    save_format="h5"
)
print("✅ Saved as trained_model_hf.h5")