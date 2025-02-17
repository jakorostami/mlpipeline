import os
import joblib
from typing import Dict

def load_ensemble_models(model_dir: str = "saved_models", version: str = None) -> Dict:
    """
    Load ensemble models with version control
    
    Args:
        model_dir: Base directory for models
        version: Specific version to load (e.g., 'v1', 'v2'). 
                If None, loads the latest version.
    """
    loaded_models = {}
    try:
        if version:
            version_path = os.path.join(model_dir, version)
            if not os.path.exists(version_path):
                raise ValueError(f"Version {version} not found in {model_dir}")
            
            model_files = {}
            for filename in os.listdir(version_path):
                if filename.endswith('.joblib'):
                    model_name = filename.split('_')[0]
                    timestamp = filename.split('_')[1].replace('.joblib', '')
                    
                    if model_name not in model_files or timestamp > model_files[model_name][1]:
                        model_files[model_name] = (filename, timestamp)
        else:
            versions = [d for d in os.listdir(model_dir) 
                       if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('v')]
            if not versions:
                raise ValueError(f"No versioned models found in {model_dir}")
            
            latest_version = sorted(versions)[-1]
            version_path = os.path.join(model_dir, latest_version)
            
            model_files = {}
            for filename in os.listdir(version_path):
                if filename.endswith('.joblib'):
                    model_name = filename.split('_')[0]
                    timestamp = filename.split('_')[1].replace('.joblib', '')
                    
                    if model_name not in model_files or timestamp > model_files[model_name][1]:
                        model_files[model_name] = (filename, timestamp)
        
        for model_name, (filename, _) in model_files.items():
            file_path = os.path.join(version_path, filename)
            loaded_models[model_name] = joblib.load(file_path)
        
        return loaded_models
        
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")   