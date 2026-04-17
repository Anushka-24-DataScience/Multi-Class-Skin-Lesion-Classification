import tensorflow as tf
from DermaCancerScan.config.configuration import ConfigurationManager
from DermaCancerScan.components.prepare_base_model import PrepareBaseModel
from DermaCancerScan import logger


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):

        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU detected: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("No GPU detected. Running on CPU.")

        config = ConfigurationManager()

        prepare_base_model_config = config.get_prepare_base_model_config()

        logger.info(
            f"Using IMAGE_SIZE: {prepare_base_model_config.params_image_size}"
        )

        prepare_base_model = PrepareBaseModel(
            config=prepare_base_model_config
        )

        logger.info("Loading EfficientNet base model...")
        prepare_base_model.get_base_model()

        logger.info("Updating base model with custom head & fine-tuning...")
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
    except Exception as e:
        logger.exception(e)
        raise e