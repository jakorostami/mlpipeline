import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from ml_pipeline import PipedriveJobPipeline

def main(version_training):
    """
    This is the manual trigger of the full training and feature engineering pipelines.
    """
    
    engine = PipedriveJobPipeline(version_id=version_training)

    engine.start_pipeline()

    return None



if __name__ == "__main__":

     
    parser = argparse.ArgumentParser(description='Training pipeline engine for transaction results. Made by Jako Rostami')
    parser.add_argument("-version_id", "--version_id", type=str, default=None, help="Versioning of your machine learning models")
    args = parser.parse_args()

    main(args.version_id)


