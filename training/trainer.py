import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    f1_score,
    classification_report
)
from scipy.stats import gmean
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, List, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from training.model_loader import ModelLoader

class ModelConfig(BaseModel):
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    n_splits: int = Field(default=5, ge=1)
    shuffle: bool = Field(default=True)
    random_state: int = Field(default=42)
    model_path: str = Field(default="saved_models")
    ensemble_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class ModelMetrics(BaseModel):
    accuracy: float
    f1_score: float
    confusion_matrix: List[List[int]]
    classification_report: str

class ModelTrainer:
    """
    Training orchestrator that does the whole machine learning training setup.
    """
    def __init__(self, config: ModelConfig = None, model_path: str = None):
        self.config = config or ModelConfig()
        self.models = ModelLoader.create_models(model_path)
        self.trained_models = {}
        self.results = {}
        self.logger = self._setup_logger()
        self.ensemble_predictions = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def make_pipeline(self, classifier: Any) -> Pipeline:
        return Pipeline([('classifier', classifier)])

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        """
        Train and evaluate all models using k-fold cross validation
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple containing results dictionary and trained models dictionary
        """
        self.logger.info(f"Starting model training with {self.config.n_splits}-fold CV")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state
        )
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}")
            accuracies = []
            f1_scores = []
            best_score = -1
            best_pipeline = None
            
            for i in range(self.config.n_splits):
                X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(
                    X_train, y_train,
                    test_size=1/self.config.n_splits,
                    random_state=self.config.random_state + i
                )
                
                pipeline = self.make_pipeline(model)
                pipeline.fit(X_fold_train, y_fold_train)
                
                predictions = pipeline.predict(X_fold_val)
                fold_accuracy = accuracy_score(y_fold_val, predictions)
                fold_f1 = f1_score(y_fold_val, predictions)
                
                accuracies.append(fold_accuracy)
                f1_scores.append(fold_f1)
                
                if fold_f1 > best_score:
                    best_score = fold_f1
                    best_pipeline = pipeline
                
                self.logger.info(
                    f"{name} - Fold {i + 1}/{self.config.n_splits}: "
                    f"Accuracy = {fold_accuracy:.3f}, F1 = {fold_f1:.3f}"
                )
            
            self.trained_models[name] = best_pipeline
            
            final_predictions = best_pipeline.predict(X_test)
            
            self.results[name] = ModelMetrics(
                accuracy=accuracy_score(y_test, final_predictions),
                f1_score=f1_score(y_test, final_predictions),
                confusion_matrix=confusion_matrix(y_test, final_predictions).tolist(),
                classification_report=classification_report(y_test, final_predictions)
            )
            
            self._log_model_metrics(
                name, 
                self.results[name],
                np.mean(accuracies),
                np.std(accuracies),
                np.mean(f1_scores),
                np.std(f1_scores)
            )

        self.create_ensemble_predictions(X_test, y_test)
        
        return self.results, self.trained_models

    def create_ensemble_predictions(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        self.logger.info("Creating ensemble predictions")
        
        y_pred_probas = {}
        for name, model in self.trained_models.items():
            try:
                y_pred_probas[name] = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                self.logger.warning(f"Model {name} doesn't support predict_proba, skipping for ensemble")
                continue
        
        y_pred_proba_ensemble = gmean(np.array(list(y_pred_probas.values())), axis=0)
        
        y_pred_ensemble = (y_pred_proba_ensemble >= self.config.ensemble_threshold).astype(int)

        ensemble_metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred_ensemble),
            f1_score=f1_score(y_test, y_pred_ensemble),
            confusion_matrix=confusion_matrix(y_test, y_pred_ensemble).tolist(),
            classification_report=classification_report(y_test, y_pred_ensemble)
        )
        
        self.results['Ensemble'] = ensemble_metrics
        self._log_model_metrics('Ensemble', ensemble_metrics)
        
        self.ensemble_predictions = {
            'probabilities': y_pred_proba_ensemble,
            'predictions': y_pred_ensemble
        }
        
        return ensemble_metrics

    def _log_model_metrics(self, model_name: str, metrics: ModelMetrics, 
                          cv_accuracy_mean: float = None, cv_accuracy_std: float = None,
                          cv_f1_mean: float = None, cv_f1_std: float = None) -> None:
        cv_stats = ""
        if cv_accuracy_mean is not None:
            cv_stats = (f'\nCV Accuracy: {cv_accuracy_mean:.3f} (±{cv_accuracy_std:.3f})\n'
                       f'CV F1-score: {cv_f1_mean:.3f} (±{cv_f1_std:.3f})')
            
        self.logger.info(
            f'\n{model_name} Results:'
            f'{cv_stats}\n'
            f'Test Accuracy: {metrics.accuracy:.3f}\n'
            f'Test F1-score: {metrics.f1_score:.3f}\n'
            f'Classification Report:\n{metrics.classification_report}'
        )

    def save_models(self, version: str, base_path="saved_models") -> Dict:
        """Save the best models from cross-validation"""
        os.makedirs(base_path, exist_ok=True)
        version_path = os.path.join(base_path, version)
        os.makedirs(version_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_metadata = {
            'version': version,
            'timestamp': timestamp,
            'model_paths': {},
            'base_path': version_path,
            'cv_folds': self.config.n_splits
        }
        
        for model_name, model in self.trained_models.items():
            filename = f"{model_name}_{timestamp}.joblib"
            filepath = os.path.join(version_path, filename)
            joblib.dump(model, filepath)
            saved_metadata['model_paths'][model_name] = filepath
            
        return saved_metadata