import json
import numpy as np
from datetime import datetime
from typing import Dict
import pandas as pd
import os

class ModelMonitor:
    """
    Monitoring service for the machine learning models during inference time.
    Makes sure we log the predictions, drift, respons times and other metrics which are then
    used as decision support to retrain models or provide a live check on model quality. 
    """
    def __init__(self, 
                 metrics_file_path: str = "/logs/metrics/model_metrics.json",
                 drift_threshold: float = 0.1):
        self.metrics_file_path = metrics_file_path
        self.feature_stats = {}
        self.drift_threshold = drift_threshold
        
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        
        if not os.path.exists(metrics_file_path):
            self._initialize_metrics_file()
    
    def _initialize_metrics_file(self):
        initial_metrics = {
            "predictions": [],
            "performance": {
                "accuracy": [],
                "response_times": []
            },
            "drift_checks": [],
            "last_updated": datetime.now().isoformat()
        }
        self._save_metrics(initial_metrics)
    
    def _save_metrics(self, metrics: Dict):
        try:
            with open(self.metrics_file_path, 'w') as f:
                json.dump(metrics, f, default=str)  # Use default=str for non-serializable objects
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
    
    def _load_metrics(self) -> Dict:
        try:
            if not os.path.exists(self.metrics_file_path):
                return {
                    "predictions": [],
                    "performance": {
                        "accuracy": [],
                        "response_times": []
                    },
                    "drift_checks": [],
                    "last_updated": datetime.now().isoformat()
                }
                
            with open(self.metrics_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading metrics file: {str(e)}")
            # If file is corrupted, start fresh
            return {
                "predictions": [],
                "performance": {
                    "accuracy": [],
                    "response_times": []
                },
                "drift_checks": [],
                "last_updated": datetime.now().isoformat()
            }

    def log_prediction(self, 
                      features: Dict,
                      prediction: float,
                      response_time: float,
                      model_versions: Dict[str, str]):
        metrics = self._load_metrics()
        
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": prediction,
            "response_time": response_time,
            "model_versions": model_versions
        }
        
        metrics["predictions"].append(prediction_data)
        metrics["last_updated"] = datetime.now().isoformat()
        
        # Keep only last 1000 predictions
        metrics["predictions"] = metrics["predictions"][-1000:]
        
        self._save_metrics(metrics)

    def check_drift(self, current_features: pd.DataFrame) -> Dict:
        try:
            drift_detected = False  # Python native bool
            drift_details = {}
            
            if not self.feature_stats:
                self.feature_stats = self._compute_feature_stats(current_features)
                drift_details = {"message": "Baseline stats initialized"}
                return {"drift_detected": False, "details": drift_details}
            
            current_stats = self._compute_feature_stats(current_features)
            
            for feature in current_stats:
                if feature not in self.feature_stats:
                    continue
                    
                try:
                    baseline_mean = float(self.feature_stats[feature]['mean'])
                    current_mean = float(current_stats[feature]['mean'])
                    baseline_std = float(self.feature_stats[feature]['std'])
                    current_std = float(current_stats[feature]['std'])
                    
                    # Calculate drift metrics
                    mean_diff = float(abs(baseline_mean - current_mean) / max(baseline_mean, 1e-7))
                    std_diff = float(abs(baseline_std - current_std) / max(baseline_std, 1e-7))
                    
                    feature_drift = bool(mean_diff > self.drift_threshold or std_diff > self.drift_threshold)
                    
                    drift_details[feature] = {
                        "drift_detected": bool(feature_drift),  # Convert numpy.bool_ to Python bool
                        "mean_difference": float(mean_diff),
                        "std_difference": float(std_diff)
                    }
                    
                    if feature_drift:
                        drift_detected = True
                except Exception as e:
                    print(f"Error checking drift for feature {feature}: {str(e)}")
                    continue
            
            return {
                "drift_detected": bool(drift_detected),  # Ensure Python native bool
                "details": drift_details
            }
        except Exception as e:
            print(f"Error in drift check: {str(e)}")
            return {"drift_detected": False, "details": {"error": str(e)}}

    def _compute_feature_stats(self, df: pd.DataFrame) -> Dict:
        stats = {}
        for column in df.columns:
            try:
                if df[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    stats[column] = {
                        'mean': df[column].mean(),
                        'std': df[column].std(),
                        'quantiles': df[column].quantile([0.25, 0.5, 0.75]).to_dict()
                    }
            except Exception as e:
                print(f"Error computing stats for column {column}: {str(e)}")
                continue
        return stats

    def get_performance_metrics(self) -> Dict:
        metrics = self._load_metrics()
        recent_predictions = metrics["predictions"][-1000:]  
        
        if not recent_predictions:
            return {"status": "No recent predictions available"}
        
        response_times = [p["response_time"] for p in recent_predictions]
        
        return {
            "average_response_time": np.mean(response_times),
            "p95_response_time": np.percentile(response_times, 95),
            "prediction_rate": len(recent_predictions) / (24 * 60 * 60),  # predictions per second
            "last_prediction_time": recent_predictions[-1]["timestamp"]
        }

    def should_retrain(self) -> tuple[bool, str]:
        metrics = self._load_metrics()
        
        if metrics["drift_checks"]:
            latest_drift = metrics["drift_checks"][-1]
            if latest_drift["drift_detected"]:
                return True, "Data drift detected"
        
        # Check performance degradation
        if len(metrics["predictions"]) >= 1000:
            recent_response_times = [p["response_time"] for p in metrics["predictions"][-1000:]]
            if np.mean(recent_response_times) > 1.0:  # More than 1 second average
                return True, "Performance degradation detected"
        
        return False, "No retraining needed"