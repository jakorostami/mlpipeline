import yaml
from typing import Dict
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

class ModelLoader:
    """
    Loading orchestrator for model experimentation. 
    """
    MODEL_CLASSES = {
        'XGBClassifier': XGBClassifier,
        'XGBRFClassifier': XGBRFClassifier,
        'CatBoostClassifier': CatBoostClassifier,
        'LogisticRegression': LogisticRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'MLPClassifier': MLPClassifier
    }

    @classmethod
    def load_model_config(cls, config_path: str) -> Dict:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @classmethod
    def create_models(cls, config_path: str) -> Dict:
        config = cls.load_model_config(config_path)
        models = {}

        for model_name, model_config in config['models'].items():
            model_type = model_config['type']
            model_params = model_config['params']

            if model_type not in cls.MODEL_CLASSES:
                print(f"Model type is: {model_type}")
                raise ValueError(f"Unknown model type: {model_type}")

            model_class = cls.MODEL_CLASSES[model_type]
            models[model_name] = model_class(**model_params)

        return models