import pytest
import os
import pandas as pd
from monitoring.monitor import ModelMonitor

os.makedirs("logs/test_logs/metrics", exist_ok=True)

@pytest.fixture
def test_monitor():
    return ModelMonitor(metrics_file_path="logs/test_logs/metrics/test_metrics.json")

def test_model_monitor_initialization(test_monitor):
    assert test_monitor.drift_threshold == 0.1
    assert hasattr(test_monitor, 'feature_stats')
    assert os.path.exists(test_monitor.metrics_file_path)

def test_model_monitor_drift_check(test_monitor):
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    
    drift_result = test_monitor.check_drift(sample_data)
    assert isinstance(drift_result, dict)
    assert "drift_detected" in drift_result
    assert "details" in drift_result

def test_should_retrain(test_monitor):
    should_retrain, reason = test_monitor.should_retrain()
    assert isinstance(should_retrain, bool)
    assert isinstance(reason, str)

def test_get_performance_metrics(test_monitor):
    metrics = test_monitor.get_performance_metrics()
    assert isinstance(metrics, dict)