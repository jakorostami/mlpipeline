import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from inference.api import app, CustomerFeatures, PredictionResponse

client = TestClient(app)

test_features = {
    "Date": ["10/12/2013"],
    "Product": ["Shirt"],
    "Gender": ["Male"],
    "Device_Type": ["Mobile"],
    "State": ["California"],
    "City": ["Los Angles"],
    "Category": ["Clothing"],
    "Customer_Login_type": ["Guest"],
    "Delivery_Type": ["one-day deliver"],
    "Individual_Price_US$": [13.0],
    "Time": ["14:30:00"],
    "Quantity": [10]
}

@pytest.fixture
def mock_monitor():
    with patch('inference.api.ModelMonitor') as mock:
        mock_instance = Mock()
        mock_instance.should_retrain.return_value = (False, "No retraining needed")
        mock_instance.check_drift.return_value = {
            "drift_detected": False,
            "details": {}
        }
        mock.return_value = mock_instance
        yield mock_instance


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "operational"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_customer_features_validation():
    features = CustomerFeatures(**test_features)
    assert len(features.Date) == len(features.Product)
    
    invalid_data = test_features.copy()
    del invalid_data["Date"]
    with pytest.raises(ValueError):
        CustomerFeatures(**invalid_data)

def test_prediction_response_validation():
    valid_response = {
        "transaction_result": [1.0],
        "probability": [0.85],
        "model_predictions": [{"model1": 0.8, "model2": 0.9}],
        "drift_detected": False,
        "drift_details": {"feature1": {"drift": False}}
    }
    
    response = PredictionResponse(**valid_response)
    assert isinstance(response.transaction_result, list)
    assert isinstance(response.probability, list)
    assert isinstance(response.model_predictions, list)
    assert isinstance(response.drift_detected, bool)


@pytest.mark.asyncio
@patch('inference.api.ensemble_models', {'model1': Mock(), 'model2': Mock()})
async def test_predict_endpoint():
    for model_name in app.ensemble_models:
        app.ensemble_models[model_name].predict_proba = Mock(return_value=[[0.7, 0.3]])

    response = client.post("/predict", json=test_features)
    assert response.status_code == 200
    
    result = response.json()
    assert "transaction_result" in result
    assert "probability" in result
    assert "model_predictions" in result
    assert isinstance(result["transaction_result"], list)
    assert isinstance(result["probability"], list)
    assert isinstance(result["model_predictions"], list)

@pytest.mark.asyncio
async def test_maintenance_status(mock_monitor):
    app.model_monitor = mock_monitor
    response = client.get("/maintenance/status")
    assert response.status_code == 200
    assert "needs_retraining" in response.json()
    assert "reason" in response.json()
