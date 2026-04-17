import tensorflow as tf
from DermaCancerScan.config.configuration import ConfigurationManager
from DermaCancerScan.components.model_training import ModelTraining
from DermaCancerScan import logger

STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # GPU setup
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"GPU detected: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("No GPU detected. Running on CPU — training will be slow.")

        config = ConfigurationManager()
        training_config = config.get_training_config()

        trainer = ModelTraining(config=training_config)

        logger.info("Step 1/4: Loading base model...")
        trainer.get_base_model()

        logger.info("Step 2/4: Creating data generators...")
        trainer.train_valid_generator()

        logger.info("Step 3/4: Compiling model...")
        trainer.compile_model()

        logger.info("Step 4/4: Training...")
        trainer.train()


if __name__ == "__main__":
    try:
        logger.info("*" * 50)
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info("x" * 50)
    except Exception as e:
        logger.exception(e)
        raise e