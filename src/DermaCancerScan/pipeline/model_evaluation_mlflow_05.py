from DermaCancerScan.config.configuration import ConfigurationManager
from DermaCancerScan.components.model_evaluation_mlflow import ModelEvaluation
from DermaCancerScan import logger

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        evaluator = ModelEvaluation(config=eval_config)

        logger.info("Step 1/3: Running evaluation...")
        evaluator.evaluate()

        logger.info("Step 2/3: Logging to MLflow...")
        evaluator.log_into_mlflow()

        logger.info("Step 3/3: Done.")
        logger.info(f"Accuracy      : {evaluator.scores['accuracy']}")
        logger.info(f"Weighted F1   : {evaluator.scores['weighted_f1']}")
        logger.info(f"Macro AUC     : {evaluator.scores['macro_auc']}")
        logger.info(f"Melanoma Recall: {evaluator.scores['melanoma_recall']}")


if __name__ == "__main__":
    try:
        logger.info("*" * 50)
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info("x" * 50)
    except Exception as e:
        logger.exception(e)
        raise e