import os
import logging

import numpy as np
import pandas as pd
import polars as pl


from preprocessing.featureprocessor import Preprocessor
from training.trainer import ModelTrainer, ModelConfig


class PipedriveJobPipeline:
    """
    Orchestrate the machine learning model training across modules.

    """

    def __init__(self,
                 training_data : str = "test_task_data.csv",
                 training_config = ModelConfig(test_size=0.2,
                                               n_splits=5,
                                               shuffle=True,
                                               random_state=42,
                                               model_path="saved_models",
                                               ensemble_threshold=0.5),
                 version_id: str = None
                 ):
        self.feature_processor = Preprocessor(training_data, inference=False, inference_input=pl.LazyFrame())
        self.training_config = training_config
        self.training_engine = ModelTrainer(self.training_config, "model_card.yaml")
        self.version_id = version_id
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


    def start_pipeline(self):
        self.logger.info(f"Training pipeline started.")
        self.logger.info(f"Feature Engineering pipeline running...")

        df = self.feature_processor.feature_engineering(date_feature="Date")

        self.logger.info(f"Feature Engineering finished.")

        self.logger.info(f"Splitting data into train and test for training...")

        X = df.drop(columns=["Transaction_id", "customer_id", "Date", "Transaction_Result"]).copy()

        Y = df["Transaction_Result"].copy()

        self.logger.info(f"Split finished.")

        self.logger.info(f"Model training initiated.")
        empty1, empty2 = self.training_engine.train_and_evaluate(X, Y)

        self.logger.info(f"All models finished training.")
        
        self.logger.info(f"Saving models...")

        self.training_engine.save_models(version=self.version_id)
        
        self.logger.info(f"Models saved!")

        self.logger.info(f"Training pipeline engine finished.")
