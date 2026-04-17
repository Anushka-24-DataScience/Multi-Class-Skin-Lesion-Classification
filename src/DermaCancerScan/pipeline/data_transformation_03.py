from DermaCancerScan.config.configuration import ConfigurationManager
from DermaCancerScan.components.data_transformation import DataTransformation
from DermaCancerScan import logger

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(
            config=data_transformation_config
        )

        # Runs generators and saves class_indices.json to artifacts/data_transformation
        data_transformation.get_data_generators()


if __name__ == "__main__":
    try:
        logger.info("*" * 50)
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info("x" * 50)
    except Exception as e:
        logger.exception(e)
        raise e