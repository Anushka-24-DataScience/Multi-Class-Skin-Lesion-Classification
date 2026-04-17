from DermaCancerScan import logger
from DermaCancerScan.pipeline.data_ingestion_01 import DataIngestionTrainingPipeline
from DermaCancerScan.pipeline.prepare_base_model_02 import PrepareBaseModelTrainingPipeline
#from DermaCancerScan.pipeline.model_training_03 import ModelTrainingPipeline
# from DermaCancerScan.pipeline.stage_04_model_evaluation import EvaluationPipeline




STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




