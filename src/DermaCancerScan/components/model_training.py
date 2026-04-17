import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from DermaCancerScan import logger
from DermaCancerScan.entity.config_entity import TrainingConfig


class ModelTraining:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.class_weights = None

    def get_base_model(self):
        model_path = str(self.config.updated_base_model_path)
        logger.info(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `dvc repro prepare_base_model` first."
            )

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        logger.info("Base model loaded successfully.")

    def train_valid_generator(self):
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:2],  # (380, 380)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

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
                validation_split=0.20
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                validation_split=0.20
            )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.20
        )

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            seed=42,
            **dataflow_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            seed=42,
            **dataflow_kwargs
        )

        # Compute class weights to handle HAM10000 imbalance
        # NV class dominates at ~67% — class weights penalize majority class
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes
        )
        self.class_weights = dict(enumerate(class_weights_array))

        logger.info(f"Train samples     : {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.valid_generator.samples}")
        logger.info(f"Class mapping     : {self.train_generator.class_indices}")
        logger.info(f"Class weights     : {self.class_weights}")

    def compile_model(self, learning_rate: float = None):
        lr = learning_rate or self.config.params_learning_rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        logger.info(f"Model compiled with LR: {lr}")

    @staticmethod
    def _prepare_callbacks(
        callbacks_dir: Path,
        early_stopping_patience: int,
        reduce_lr_patience: int,
        reduce_lr_factor: float,
        min_lr: float,
        phase: str = "phase1"   # track which phase in tensorboard
    ):
        os.makedirs(callbacks_dir, exist_ok=True)

        checkpoint_path = str(callbacks_dir / "best_model.keras")
        tensorboard_log_dir = str(callbacks_dir / "tensorboard_logs" / phase)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )

        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1
        )

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        )

        return [checkpoint_cb, tensorboard_cb, early_stopping_cb, reduce_lr_cb]

    def _unfreeze_top_layers(self):
        """Unfreeze last N layers of EfficientNetB4 for fine-tuning."""
        n = self.config.params_fine_tune_layers
        for layer in self.model.layers[-n:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                # Keep BN frozen — unfreezing BN during fine-tuning
                # destroys learned statistics and hurts performance
                layer.trainable = True

        trainable_count = sum(
            1 for l in self.model.layers if l.trainable
        )
        logger.info(f"Unfroze last {n} layers — {trainable_count} total trainable layers")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(str(path))
        logger.info(f"Model saved at: {path}")

    def train(self):
        steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        callbacks_dir = Path(self.config.root_dir) / "callbacks"

        # ── Phase 1: Train custom head only (frozen EfficientNetB4) ──
        logger.info("=" * 50)
        logger.info(f"Phase 1: Training custom head — {self.config.params_epochs} epochs")
        logger.info(f"All EfficientNetB4 layers frozen")
        logger.info("=" * 50)

        phase1_callbacks = self._prepare_callbacks(
            callbacks_dir=callbacks_dir,
            early_stopping_patience=self.config.early_stopping_patience,
            reduce_lr_patience=self.config.reduce_lr_patience,
            reduce_lr_factor=self.config.reduce_lr_factor,
            min_lr=self.config.min_lr,
            phase="phase1"
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=validation_steps,
            callbacks=phase1_callbacks,
            class_weight=self.class_weights,   # fixes class imbalance
            verbose=1
        )
        logger.info("Phase 1 complete.")

        # ── Phase 2: Fine-tune last N layers ──
        logger.info("=" * 50)
        logger.info(f"Phase 2: Fine-tuning last {self.config.params_fine_tune_layers} layers")
        logger.info(f"Fine-tune LR: {self.config.params_fine_tune_lr}")
        logger.info(f"Fine-tune epochs: {self.config.params_fine_tune_epochs}")
        logger.info("=" * 50)

        self._unfreeze_top_layers()

        # Recompile with very low LR for fine-tuning
        self.compile_model(learning_rate=self.config.params_fine_tune_lr)

        phase2_callbacks = self._prepare_callbacks(
            callbacks_dir=callbacks_dir,
            early_stopping_patience=self.config.early_stopping_patience,
            reduce_lr_patience=self.config.reduce_lr_patience,
            reduce_lr_factor=self.config.reduce_lr_factor,
            min_lr=self.config.min_lr,
            phase="phase2"
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_fine_tune_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=validation_steps,
            callbacks=phase2_callbacks,
            class_weight=self.class_weights,   # keep class weights in phase 2
            verbose=1
        )
        logger.info("Phase 2 fine-tuning complete.")

        # Save final trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )