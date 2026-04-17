import numpy as np
import tensorflow as tf
from pathlib import Path
from DermaCancerScan import logger
from DermaCancerScan.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB4(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        # Save ONLY weights for base model — avoids EagerTensor JSON bug
        # entirely. We rebuild the architecture from code during update_base_model.
        self.model.save_weights(str(self.config.base_model_path) + ".weights.h5")
        logger.info(f"Base model weights saved at: {self.config.base_model_path}.weights.h5")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.4)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.3)(x)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        # Rebuild EfficientNetB4 architecture fresh and reload weights
        # This avoids any stale EagerTensor state from the first load
        base_model = tf.keras.applications.EfficientNetB4(
            input_shape=self.config.params_image_size,
            weights=None,           # no imagenet download again
            include_top=self.config.params_include_top
        )
        base_model.load_weights(str(self.config.base_model_path) + ".weights.h5")
        logger.info("Base model weights reloaded successfully.")

        self.full_model = self._prepare_full_model(
            model=base_model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        # Full model with custom head saves fine in .keras format
        self.full_model.save(str(self.config.updated_base_model_path))
        logger.info(f"Updated model saved at: {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(str(path))
        logger.info(f"Model saved at: {path}")