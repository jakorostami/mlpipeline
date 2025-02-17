from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import numpy as np
from typing import Dict, Optional, List
import uvicorn
import polars as pl
from preprocessing.featureprocessor import Preprocessor
from scipy.stats import gmean
from utils.load_models import load_ensemble_models
from monitoring.monitor import ModelMonitor
from ml_pipeline import PipedriveJobPipeline
import os

app = FastAPI(
    title = "ML Transaction model API",
    description = "API for transaction results",
    version = "1.0"
)

ensemble_models: Dict = {}
model_monitor = None

class CustomerFeatures(BaseModel):
    Date: List[str]
    Product: List[str]
    Gender: List[str]
    Device_Type: List[str]
    State: List[str]
    City: List[str]
    Category: List[str]
    Customer_Login_type: List[str]
    Delivery_Type: List[str]
    Individual_Price_US: List[float] = Field(alias="Individual_Price_US$")
    Time: List[str]
    Quantity: List[int]

    class Config:
        allow_population_by_field_name = True 

class PredictionResponse(BaseModel):
    transaction_result: List[float]
    probability: List[float]
    model_predictions: List[Dict[str, float]]
    drift_detected: Optional[bool] = None
    drift_details: Optional[Dict] = None

class ModelVersion(BaseModel):
    version: str

@app.post("/models/load_version")
async def load_specific_version(version_info: ModelVersion):
    """Endpoint to load specific model version"""
    global ensemble_models
    try:
        ensemble_models = load_ensemble_models(version=version_info.version)
        return {
            "status": "success",
            "version": version_info.version,
            "models_loaded": len(ensemble_models),
            "model_names": list(ensemble_models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading version: {str(e)}")


@app.on_event("startup")
async def startup_event():
    global ensemble_models, model_monitor
    try:
        ensemble_models = load_ensemble_models()    # Loads latest version
        model_monitor = ModelMonitor()
        print(f"Loaded {len(ensemble_models)} ensemble models successfully")
    except Exception as e:
        print(f"Error loading ensemble models: {str(e)}")
        raise RuntimeError("Failed to load models at startup")

@app.get("/")
async def root():
    return {
        "status": "operational",
        "version": "1.0",
        "models_loaded": len(ensemble_models)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    start_time = time.time()
    drift_detected = None
    drift_details = None
    try:
        # input features
        feature_dict = features.dict(by_alias=True)
        inference_data = pl.LazyFrame(feature_dict)
        Preproc = Preprocessor(input_data_path="", inference=True, inference_input=inference_data)

        inference_df = Preproc._inference_wrangle(date_feature="Date")
        # predictions = model.predict(inference_df)

        transaction_results, probabilities, all_preds = [], [], []
        
        for row in range(len(inference_df)):
            row_preds = {}
            row_probs = []

            for name, model in ensemble_models.items():
                proba = model.predict_proba(inference_df[row:row+1])
                if proba.ndim > 1:
                    row_probs.append( float(proba[0, 1]) )
                else:
                    row_probs.append( float(proba[0]) )
                row_preds[name] = row_probs[-1]

            ensemble_proba = float(gmean(np.array(row_probs)))
            probabilities.append(ensemble_proba)
            threshold = 0.5
            final_prediction = float(ensemble_proba >= threshold)
            transaction_results.append(final_prediction)
            all_preds.append(row_preds)

        for j in range(len(inference_df)):
            model_monitor.log_prediction(
            features = features.dict(),
            prediction = transaction_results[j],
            response_time = time.time() - start_time,
            model_versions = {name: getattr(model, "version", "unknown") for name, model in ensemble_models.items()}
        )

        # # Get predictions from each model
        # model_predictions = {}
        # for name, model in ensemble_models.items():
        #     proba = model.predict_proba(inference_df)
        #     if proba.ndim > 1:
        #         model_predictions[name] = float(proba[0, 1])  # Get probability of positive class
        #     else:
        #         model_predictions[name] = float(proba[0])  # If only one probability is returned

        # Combine probabilities using geometric mean
        # ensemble_proba = float(gmean(np.array(list(model_predictions.values()))))

        # Get final prediction
        # threshold = 0.5
        # final_prediction = float(ensemble_proba >= threshold)
        

        # Start monitoring
        # model_monitor.log_prediction(
        #     features = features.dict(),
        #     prediction = final_prediction,
        #     response_time = time.time() - start_time,
        #     model_versions = {name: getattr(model, "version", "unknown") for name, model in ensemble_models.items()}
        # )

        drift_check = model_monitor.check_drift(inference_df)
        drift_detected = drift_check["drift_detected"]
        drift_details = drift_check["details"]

        return {
            "transaction_result": transaction_results,
            "probability": probabilities,
            "model_predictions": all_preds,
            "drift_detected": drift_detected,
            "drift_details": drift_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    model_info = {}
    for name, _ in ensemble_models.items():
        model_info[name] = {
            "loaded": True,
            "timestamp": os.path.getmtime(os.path.join("saved_models", f"{name}_*.joblib"))
        }
    
    return {
        "status": "healthy",
        "models_loaded": len(ensemble_models),
        "model_details": model_info
    }

@app.post("/models/reload")
async def reload_models(version_info: ModelVersion):
    """Endpoint to reload all models"""
    global ensemble_models
    try:
        ensemble_models = load_ensemble_models(version = version_info.version)
        return {
            "status": "success",
            "models_loaded": len(ensemble_models),
            "model_names": list(ensemble_models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")
    
@app.get("/maintenance/status")
async def check_maintenance_status():
    should_retrain, reason = model_monitor.should_retrain()
    return {
        "needs_retraining": should_retrain,
        "reason": reason
    }

@app.get("/maintenance/metrics")
async def get_performance_metrics():
    return model_monitor.get_performance_metrics()

@app.post("/maintenance/retrain")
async def retrain_models(version_info: ModelVersion):
    try:
        current_version = version_info.version
        new_version = f"v{int(current_version[1:]) + 1}" if current_version else "v1"
        
        # Use existing pipeline
        engine = PipedriveJobPipeline(version_id=new_version)
        engine.start_pipeline()
        
        # Reload models after training
        global ensemble_models
        ensemble_models = load_ensemble_models(version=new_version)
        
        return {"status": "success", "new_version": new_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)