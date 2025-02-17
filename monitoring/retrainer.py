import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import joblib
import os

from training.trainer import ModelTrainer, ModelConfig

class ModelRetrainer:
    """
    A retraining module that does retraining and versioning of the models.
    Triggers the training pipeline via ModelTrainer. 
    """
    def __init__(self, 
                 model_dir: str = "/saved_models",
                 data_dir: str = "/data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

    def save_training_data(self, features: pd.DataFrame, labels: pd.Series):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features.to_csv(f"{self.data_dir}/features_{timestamp}.csv", index=False)
        labels.to_csv(f"{self.data_dir}/labels_{timestamp}.csv", index=False)

    def retrain_models(self, 
                      new_features: pd.DataFrame, 
                      new_labels: pd.Series,
                      version: str) -> Dict:
        version_dir = os.path.join(self.model_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trainer = ModelTrainer(ModelConfig())
        results, trained_models = trainer.train_and_evaluate(new_features, new_labels)
        
        for name, model in trained_models.items():
            model_path = os.path.join(version_dir, f"{name}_{timestamp}.joblib")
            joblib.dump(model, model_path)
        
        metadata = {
            "version": version,
            "timestamp": timestamp,
            "metrics": results,
            "data_shape": new_features.shape
        }
        
        with open(os.path.join(version_dir, f"metadata_{timestamp}.json"), 'w') as f:
            json.dump(metadata, f)
        
        return metadata

    def get_latest_version(self) -> Optional[str]:
        if not os.path.exists(self.model_dir):
            return None
            
        versions = [d for d in os.listdir(self.model_dir) 
                   if os.path.isdir(os.path.join(self.model_dir, d))]
        
        if not versions:
            return None
            
        return sorted(versions)[-1]