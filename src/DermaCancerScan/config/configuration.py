from DermaCancerScan.constants import *
import os
from pathlib import Path

from DermaCancerScan.utils.common import read_yaml, create_directories
from DermaCancerScan.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    DataTransformationConfig,TrainingConfig, EvaluationConfig
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create root artifacts directory
        create_directories([self.config.artifacts_root])

    # ==============================
    # Data Ingestion Configuration
    # ==============================
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

   
    # def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
    #     config = self.config.prepare_base_model
    #     params = self.params.prepare_base_model  # Access nested parameters

    #     create_directories([config.root_dir])

    #     prepare_base_model_config = PrepareBaseModelConfig(
    #         root_dir=Path(config.root_dir),
    #         base_model_path=Path(config.base_model_path),
    #         updated_base_model_path=Path(config.updated_base_model_path),
    #         params_image_size=params.IMAGE_SIZE,
    #         params_learning_rate=params.LEARNING_RATE,
    #         params_include_top=params.INCLUDE_TOP,
    #         params_weights=params.WEIGHTS,
    #         params_classes=params.CLASSES
    #     )

    #     return prepare_base_model_config
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),           # now has .keras extension
            updated_base_model_path=Path(config.updated_base_model_path),  # now has .keras extension
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES
        )

        return prepare_base_model_config
    # ==================================
    # Data Transformation Configuration
    # ==================================
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_dir=Path(config.train_dir),
            test_dir=Path(config.test_dir),
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
        )

        return data_transformation_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params.training
        callbacks = self.params.callbacks

        training_data = Path(self.config.data_ingestion.unzip_dir) / "data" / "train"

        if not training_data.exists():
            raise FileNotFoundError(
                f"Training data not found at: {training_data}"
            )

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,

            # Fine-tuning
            params_fine_tune_epochs=params.FINE_TUNE_EPOCHS,
            params_fine_tune_lr=params.FINE_TUNE_LR,
            params_fine_tune_layers=params.FINE_TUNE_LAYERS,

            # Callbacks
            early_stopping_patience=callbacks.EARLY_STOPPING_PATIENCE,
            reduce_lr_patience=callbacks.REDUCE_LR_PATIENCE,
            reduce_lr_factor=callbacks.REDUCE_LR_FACTOR,
            min_lr=callbacks.MIN_LR
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        params = self.params.training

        create_directories([Path(evaluation.root_dir)])

        return EvaluationConfig(
            root_dir=Path(evaluation.root_dir),
            trained_model_path=Path(evaluation.trained_model_path),
            test_data=Path(evaluation.test_data),
            mlflow_uri=self.config.mlflow.tracking_uri,
            all_params=dict(self.params),
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_learning_rate=params.LEARNING_RATE,
            params_is_augmentation=params.AUGMENTATION
        )