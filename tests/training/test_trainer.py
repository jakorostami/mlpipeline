import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from training.trainer import ModelTrainer, ModelConfig

@pytest.fixture
def trainer():
    config = ModelConfig(
        test_size=0.2,
        n_splits=5,
        random_state=42
    )
    return ModelTrainer(config=config, model_path="tests/test_model_card.yaml")

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_trainer_initialization(trainer):
    assert trainer.config.test_size == 0.2
    assert trainer.config.n_splits == 5
    assert trainer.config.random_state == 42
    assert hasattr(trainer, 'logger')

def test_make_pipeline(trainer):
    mock_classifier = Mock()
    pipeline = trainer.make_pipeline(mock_classifier)
    assert 'classifier' in pipeline.named_steps

@patch('training.model_loader.ModelLoader.create_models')
def test_train_and_evaluate(mock_create_models, trainer, sample_data):
    X, y = sample_data
    
    mock_model = Mock()
    mock_model.predict.return_value = np.zeros(len(y))
    mock_model.predict_proba.return_value = np.random.rand(len(y), 2)
    
    mock_create_models.return_value = {'model1': mock_model}
    
    results, trained_models = trainer.train_and_evaluate(X, y)
    
    assert 'AdaBoost' in results
    assert 'AdaBoost' in trained_models
    assert 'accuracy' in results['AdaBoost'].dict()
    assert 'f1_score' in results['AdaBoost'].dict()

def test_model_config_validation():
    valid_config = ModelConfig(test_size=0.2, n_splits=5)
    assert valid_config.test_size == 0.2
    
    with pytest.raises(ValueError):
        ModelConfig(test_size=1.5)