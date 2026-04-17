import json
import tensorflow as tf
from pathlib import Path
from DermaCancerScan import logger


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def get_data_generators(self):
        """
        Creates and returns train/valid/test generators.
        Called directly by the training pipeline.
        No rescaling — EfficientNetB4 has built-in preprocessing.
        """
        img_size = tuple(self.config.params_image_size)  # (380, 380)
        batch_size = self.config.params_batch_size

        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode="nearest",
                validation_split=0.2
                # No rescale — EfficientNetB4 handles preprocessing internally
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                validation_split=0.2
                # No rescale — EfficientNetB4 handles preprocessing internally
            )

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        # No rescale for test either

        train_generator = train_datagen.flow_from_directory(
            directory=self.config.train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=42
        )

        valid_generator = train_datagen.flow_from_directory(
            directory=self.config.train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=42
        )

        test_generator = test_datagen.flow_from_directory(
            directory=self.config.test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )

        # Save class indices to disk — this gives DVC a real output to track
        # and lets you verify class mapping during evaluation
        class_indices_path = Path(self.config.root_dir) / "class_indices.json"
        with open(class_indices_path, "w") as f:
            json.dump(train_generator.class_indices, f, indent=4)
        logger.info(f"Class indices saved at: {class_indices_path}")
        logger.info(f"Class mapping: {train_generator.class_indices}")
        logger.info(f"Train samples   : {train_generator.samples}")
        logger.info(f"Validation samples: {valid_generator.samples}")
        logger.info(f"Test samples    : {test_generator.samples}")

        return train_generator, valid_generator, test_generator