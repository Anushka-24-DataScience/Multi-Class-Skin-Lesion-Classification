# import json
# import tempfile
# import numpy as np
# import tensorflow as tf
# from pathlib import Path
# import mlflow
# import mlflow.keras
# from urllib.parse import urlparse
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     f1_score
# )
# import matplotlib.pyplot as plt
# import seaborn as sns

# from DermaCancerScan import logger
# from DermaCancerScan.entity.config_entity import EvaluationConfig
# from DermaCancerScan.utils.common import save_json
# import tempfile
# import os


# # HAM10000 class names in standard order
# HAM10000_CLASSES = [
#     "akiec",   # Actinic Keratoses
#     "bcc",     # Basal Cell Carcinoma
#     "bkl",     # Benign Keratosis
#     "df",      # Dermatofibroma
#     "mel",     # Melanoma  ← most critical to detect
#     "nv",      # Melanocytic Nevi
#     "vasc"     # Vascular Lesions
# ]


# class ModelEvaluation:
#     def __init__(self, config: EvaluationConfig):
#         self.config = config
#         self.model = None
#         self.test_generator = None
#         self.score = None
#         self.y_true = None
#         self.y_pred = None
#         self.y_pred_probs = None

#     @staticmethod
#     def load_model(path: Path) -> tf.keras.Model:
#         model = tf.keras.models.load_model(str(path), compile=False)
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(),
#             loss=tf.keras.losses.CategoricalCrossentropy(),
#             metrics=[
#                 "accuracy",
#                 tf.keras.metrics.Precision(name="precision"),
#                 tf.keras.metrics.Recall(name="recall"),
#                 tf.keras.metrics.AUC(name="auc")
#             ]
#         )
#         return model

#     def get_test_generator(self):
#         # No rescale — EfficientNetB4 has built-in preprocessing
#         test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

#         self.test_generator = test_datagen.flow_from_directory(
#             directory=self.config.test_data,
#             target_size=self.config.params_image_size[:2],  # (380, 380)
#             batch_size=self.config.params_batch_size,
#             class_mode="categorical",
#             shuffle=False   # CRITICAL — must be False for correct evaluation
#         )
#         logger.info(f"Test samples  : {self.test_generator.samples}")
#         logger.info(f"Class mapping : {self.test_generator.class_indices}")

#     def evaluate(self):
#         self.model = self.load_model(self.config.trained_model_path)
#         self.get_test_generator()

#         # Standard keras metrics
#         self.score = self.model.evaluate(self.test_generator, verbose=1)

#         # Predictions for sklearn metrics
#         self.y_pred_probs = self.model.predict(self.test_generator, verbose=1)
#         self.y_pred = np.argmax(self.y_pred_probs, axis=1)
#         self.y_true = self.test_generator.classes
#         class_names = list(self.test_generator.class_indices.keys())

#         # Per-class metrics
#         report = classification_report(
#             self.y_true, self.y_pred,
#             target_names=class_names,
#             output_dict=True
#         )

#         # Weighted F1 — best single metric for imbalanced skin cancer data
#         weighted_f1 = f1_score(self.y_true, self.y_pred, average="weighted")

#         # Macro AUC — one-vs-rest for multiclass
#         macro_auc = roc_auc_score(
#             self.y_true,
#             self.y_pred_probs,
#             multi_class="ovr",
#             average="macro"
#         )

#         # Melanoma-specific recall — most critical clinical metric
#         mel_idx = class_names.index("mel") if "mel" in class_names else None
#         mel_recall = report["mel"]["recall"] if mel_idx is not None else None

#         scores = {
#             # Overall metrics
#             "loss"          : round(float(self.score[0]), 4),
#             "accuracy"      : round(float(self.score[1]), 4),
#             "precision"     : round(float(self.score[2]), 4),
#             "recall"        : round(float(self.score[3]), 4),
#             "auc"           : round(float(self.score[4]), 4),
#             "weighted_f1"   : round(float(weighted_f1), 4),
#             "macro_auc"     : round(float(macro_auc), 4),

#             # Melanoma-specific — most clinically important
#             "melanoma_recall"   : round(float(mel_recall), 4) if mel_recall else None,
#             "melanoma_precision": round(float(report["mel"]["precision"]), 4) if mel_idx is not None else None,
#             "melanoma_f1"       : round(float(report["mel"]["f1-score"]), 4) if mel_idx is not None else None,

#             # Per-class F1 for all 7 classes
#             "per_class_f1": {
#                 cls: round(report[cls]["f1-score"], 4)
#                 for cls in class_names
#             }
#         }

#         logger.info(f"\n{classification_report(self.y_true, self.y_pred, target_names=class_names)}")
#         logger.info(f"Weighted F1  : {scores['weighted_f1']}")
#         logger.info(f"Macro AUC    : {scores['macro_auc']}")
#         logger.info(f"Melanoma Recall: {scores['melanoma_recall']}")

#         # Save confusion matrix
#         self._save_confusion_matrix(self.y_true, self.y_pred, class_names)

#         # Save scores to JSON
#         save_json(path=Path(self.config.root_dir) / "scores.json", data=scores)

#         self.scores = scores
#         return scores

#     def _save_confusion_matrix(self, y_true, y_pred, class_names):
#         cm = confusion_matrix(y_true, y_pred)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(
#             cm, annot=True, fmt="d",
#             xticklabels=class_names,
#             yticklabels=class_names,
#             cmap="Blues"
#         )
#         plt.title("Confusion Matrix — HAM10000 Skin Cancer Classification")
#         plt.ylabel("True Label")
#         plt.xlabel("Predicted Label")
#         plt.tight_layout()
#         cm_path = Path(self.config.root_dir) / "confusion_matrix.png"
#         plt.savefig(str(cm_path), dpi=150)
#         plt.close()
#         logger.info(f"Confusion matrix saved at: {cm_path}")

#     def log_into_mlflow(self):
#         mlflow.set_tracking_uri(self.config.mlflow_uri)
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#         with mlflow.start_run():
#             # Log all hyperparameters
#             mlflow.log_params({
#                 "image_size"        : str(self.config.params_image_size),
#                 "batch_size"        : self.config.params_batch_size,
#                 "epochs"            : self.config.params_epochs,
#                 "learning_rate"     : self.config.params_learning_rate,
#                 "augmentation"      : self.config.params_is_augmentation,
#                 "model_architecture": "EfficientNetB4",
#                 "dataset"           : "HAM10000",
#                 "num_classes"       : 7
#             })

#             # Log all metrics
#             mlflow.log_metrics({
#                 "test_loss"         : self.scores["loss"],
#                 "test_accuracy"     : self.scores["accuracy"],
#                 "test_precision"    : self.scores["precision"],
#                 "test_recall"       : self.scores["recall"],
#                 "test_auc"          : self.scores["auc"],
#                 "weighted_f1"       : self.scores["weighted_f1"],
#                 "macro_auc"         : self.scores["macro_auc"],
#                 "melanoma_recall"   : self.scores["melanoma_recall"],
#                 "melanoma_precision": self.scores["melanoma_precision"],
#                 "melanoma_f1"       : self.scores["melanoma_f1"],
#             })

#             # Log per-class F1 scores individually
#             for cls, f1 in self.scores["per_class_f1"].items():
#                 mlflow.log_metric(f"f1_{cls}", f1)

#             # Log confusion matrix as artifact
#             mlflow.log_artifact(
#                 str(Path(self.config.root_dir) / "confusion_matrix.png")
#             )

#             # Log scores JSON as artifact
#             mlflow.log_artifact(
#                 str(Path(self.config.root_dir) / "scores.json")
#             )

#             # Register model
#             if tracking_url_type_store != "file":
#                mlflow.keras.log_model(
#                     self.model,"model",
#         registered_model_name="EfficientNetB4_SkinCancer_HAM10000"
#     )
#             else:
#                 model_path = "model.keras"
#                 self.model.save(model_path)
#                 mlflow.log_artifact(model_path, artifact_path="model")









            
            # if tracking_url_type_store != "file":
            #     mlflow.keras.log_model(
            #         self.model, "model",
            #         registered_model_name="EfficientNetB4_SkinCancer_HAM10000"
            #     )
            # # else:
            # #     mlflow.keras.log_model(self.model,artifact_path="model",keras_model_kwargs={"save_format": "keras"}
            # #    )
            # #     # mlflow.keras.log_model(self.model, "model")

            # # logger.info("MLflow logging completed.")
  

            # else:
            #     with tempfile.TemporaryDirectory() as tmp_dir:
            #         model_path = os.path.join(tmp_dir, "model.keras")  # ✅ required extension

            #         # Save model manually
            #         self.model.save(model_path)

            #         # Log as artifact
            #         mlflow.log_artifact(model_path, artifact_path="model")


import json
import tempfile
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from DermaCancerScan import logger
from DermaCancerScan.entity.config_entity import EvaluationConfig
from DermaCancerScan.utils.common import save_json


# HAM10000 class names in standard order
HAM10000_CLASSES = [
    "akiec",   # Actinic Keratoses
    "bcc",     # Basal Cell Carcinoma
    "bkl",     # Benign Keratosis
    "df",      # Dermatofibroma
    "mel",     # Melanoma  ← most critical to detect
    "nv",      # Melanocytic Nevi
    "vasc"     # Vascular Lesions
]


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.test_generator = None
        self.score = None
        self.y_true = None
        self.y_pred = None
        self.y_pred_probs = None
        self.scores = None

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        model = tf.keras.models.load_model(str(path), compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        return model

    def get_test_generator(self):
        # No rescale — EfficientNetB4 has built-in preprocessing
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        self.test_generator = test_datagen.flow_from_directory(
            directory=self.config.test_data,
            target_size=self.config.params_image_size[:2],  # (380, 380)
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            shuffle=False   # CRITICAL — must be False for correct evaluation
        )
        logger.info(f"Test samples  : {self.test_generator.samples}")
        logger.info(f"Class mapping : {self.test_generator.class_indices}")

    def evaluate(self):
        self.model = self.load_model(self.config.trained_model_path)
        self.get_test_generator()

        # Standard keras metrics
        self.score = self.model.evaluate(self.test_generator, verbose=1)

        # Predictions for sklearn metrics
        self.y_pred_probs = self.model.predict(self.test_generator, verbose=1)
        self.y_pred = np.argmax(self.y_pred_probs, axis=1)
        self.y_true = self.test_generator.classes
        class_names = list(self.test_generator.class_indices.keys())

        # Per-class metrics
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=class_names,
            output_dict=True
        )

        # Weighted F1 — best single metric for imbalanced skin cancer data
        weighted_f1 = f1_score(self.y_true, self.y_pred, average="weighted")

        # Macro AUC — one-vs-rest for multiclass
        macro_auc = roc_auc_score(
            self.y_true,
            self.y_pred_probs,
            multi_class="ovr",
            average="macro"
        )

        # Melanoma-specific recall — most critical clinical metric
        mel_idx = class_names.index("mel") if "mel" in class_names else None
        mel_recall = report["mel"]["recall"] if mel_idx is not None else None

        scores = {
            # Overall metrics
            "loss"          : round(float(self.score[0]), 4),
            "accuracy"      : round(float(self.score[1]), 4),
            "precision"     : round(float(self.score[2]), 4),
            "recall"        : round(float(self.score[3]), 4),
            "auc"           : round(float(self.score[4]), 4),
            "weighted_f1"   : round(float(weighted_f1), 4),
            "macro_auc"     : round(float(macro_auc), 4),

            # Melanoma-specific — most clinically important
            "melanoma_recall"   : round(float(mel_recall), 4) if mel_recall else None,
            "melanoma_precision": round(float(report["mel"]["precision"]), 4) if mel_idx is not None else None,
            "melanoma_f1"       : round(float(report["mel"]["f1-score"]), 4) if mel_idx is not None else None,

            # Per-class F1 for all 7 classes
            "per_class_f1": {
                cls: round(report[cls]["f1-score"], 4)
                for cls in class_names
            }
        }

        logger.info(f"\n{classification_report(self.y_true, self.y_pred, target_names=class_names)}")
        logger.info(f"Weighted F1  : {scores['weighted_f1']}")
        logger.info(f"Macro AUC    : {scores['macro_auc']}")
        logger.info(f"Melanoma Recall: {scores['melanoma_recall']}")

        # Save confusion matrix
        self._save_confusion_matrix(self.y_true, self.y_pred, class_names)

        # Save scores to JSON
        save_json(path=Path(self.config.root_dir) / "scores.json", data=scores)

        self.scores = scores
        return scores

    def _save_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues"
        )
        plt.title("Confusion Matrix — HAM10000 Skin Cancer Classification")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = Path(self.config.root_dir) / "confusion_matrix.png"
        plt.savefig(str(cm_path), dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved at: {cm_path}")

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # ── Log all hyperparameters ──
            mlflow.log_params({
                "image_size"        : str(self.config.params_image_size),
                "batch_size"        : self.config.params_batch_size,
                "epochs"            : self.config.params_epochs,
                "learning_rate"     : self.config.params_learning_rate,
                "augmentation"      : self.config.params_is_augmentation,
                "model_architecture": "EfficientNetB4",
                "dataset"           : "HAM10000",
                "num_classes"       : 7
            })

            # ── Log all scalar metrics ──
            mlflow.log_metrics({
                "test_loss"         : self.scores["loss"],
                "test_accuracy"     : self.scores["accuracy"],
                "test_precision"    : self.scores["precision"],
                "test_recall"       : self.scores["recall"],
                "test_auc"          : self.scores["auc"],
                "weighted_f1"       : self.scores["weighted_f1"],
                "macro_auc"         : self.scores["macro_auc"],
                "melanoma_recall"   : self.scores["melanoma_recall"],
                "melanoma_precision": self.scores["melanoma_precision"],
                "melanoma_f1"       : self.scores["melanoma_f1"],
            })

            # ── Log per-class F1 scores ──
            for cls, f1 in self.scores["per_class_f1"].items():
                mlflow.log_metric(f"f1_{cls}", f1)

            # ── Log confusion matrix + scores as artifacts ──
            mlflow.log_artifact(
                str(Path(self.config.root_dir) / "confusion_matrix.png")
            )
            mlflow.log_artifact(
                str(Path(self.config.root_dir) / "scores.json")
            )

            # ── Log model ──
            # mlflow.keras.log_model() internally saves without a .keras
            # extension which crashes on Keras 3.x. Fix: save to a temp
            # .keras file manually and log as artifact instead.
            if tracking_url_type_store != "file":
                # Remote tracking (DagsHub / MLflow server)
                # Save to temp file with explicit .keras extension
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_model_path = os.path.join(tmp_dir, "model.keras")
                    self.model.save(tmp_model_path)
                    mlflow.log_artifact(
                        tmp_model_path,
                        artifact_path="model"
                    )
                logger.info("Model logged to MLflow (remote) as model.keras artifact")
            else:
                # Local file tracking — save directly to artifacts/evaluation/
                local_model_path = str(
                    Path(self.config.root_dir) / "model.keras"
                )
                self.model.save(local_model_path)
                mlflow.log_artifact(local_model_path, artifact_path="model")
                logger.info(f"Model logged to MLflow (local) at: {local_model_path}")