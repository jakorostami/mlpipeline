import pytest
import pandas as pd
import os

from monitoring.retrainer import ModelRetrainer

@pytest.fixture
def retrainer():
    return ModelRetrainer(model_dir="saved_models/test_models", data_dir="data/test_data")

def test_retrainer_initialization(retrainer):
    assert os.path.exists(retrainer.model_dir)
    assert os.path.exists(retrainer.data_dir)

def test_save_training_data(retrainer):
    features = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    labels = pd.Series([0, 1])
    
    retrainer.save_training_data(features, labels)
    assert any(file.startswith('features_') for file in os.listdir(retrainer.data_dir))
    assert any(file.startswith('labels_') for file in os.listdir(retrainer.data_dir))

def test_get_latest_version(retrainer):
    os.makedirs(os.path.join(retrainer.model_dir, 'v1'), exist_ok=True)
    os.makedirs(os.path.join(retrainer.model_dir, 'v2'), exist_ok=True)
    
    latest = retrainer.get_latest_version()
    assert latest == 'v2'