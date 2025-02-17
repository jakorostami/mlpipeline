import pytest
import pandas as pd
import polars as pl
from preprocessing.featureprocessor import TimeSeriesFeatureEngineer, Preprocessor

@pytest.fixture
def time_engineer():
    return TimeSeriesFeatureEngineer(date_column='Date')

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Date': ['2024-02-16', '2024-02-17'],
        'Value': [100, 200]
    })

def test_time_series_feature_engineer_init(time_engineer):
    assert time_engineer.date_column == 'Date'
    assert isinstance(time_engineer.generated_features, list)

def test_add_basic_time_features(time_engineer, sample_data):
    result = time_engineer.add_basic_time_features(sample_data)
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'year' in result.columns

def test_add_cyclical_features(time_engineer, sample_data):
    base_features = time_engineer.add_basic_time_features(sample_data)
    result = time_engineer.add_cyclical_features(base_features)
    assert 'month_sin' in result.columns
    assert 'month_cos' in result.columns

@pytest.fixture
def preprocessor():
    sample_data = pl.LazyFrame({
        'Date': ['2024-02-16'],
        'Individual_Price_US$': ['999.99'],
        'Quantity': [1],
        'Time': ['14:30:00']
    })
    return Preprocessor(input_data_path="", inference=True, inference_input=sample_data)

def test_inference_wrangle(preprocessor):
    result = preprocessor._inference_wrangle(date_feature="Date")
    assert 'hour' in result.columns
    assert 'minute' in result.columns
    assert isinstance(result, pd.DataFrame)