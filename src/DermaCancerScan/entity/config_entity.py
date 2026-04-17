from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    params_image_size: List[int]
    params_batch_size: int
    params_is_augmentation: bool


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: List[int]
    params_learning_rate: float

    # Fine-tuning parameters
    params_fine_tune_epochs: int
    params_fine_tune_lr: float
    params_fine_tune_layers: int

    # Callback parameters
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    min_lr: float


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    test_data: Path
    mlflow_uri: str
    all_params: dict
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_is_augmentation: bool