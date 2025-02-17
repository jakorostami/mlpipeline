import pandas as pd
import polars as pl
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class TimeSeriesFeatureEngineer:
    """
    A class for engineering features from datetime data in time series analysis.
    Handles both cyclical and categorical features with a focus on temporal patterns.
    """
    def __init__(self, date_column: str = 'DT'):
        """
        Initialize the feature engineer with configuration.
        
        Args:
            date_column (str): Name of the datetime column in the dataframe
        """
        self.date_column = date_column
        self.generated_features = []
        
    def _create_cyclical_feature(self, value: pd.Series, period: int) -> tuple:
        """
        Create sine and cosine transformations for cyclical features.
        
        Args:
            value: Series containing the numeric values to transform
            period: The period of the cycle (e.g., 7 for days of week)
            
        Returns:
            tuple: (sine transformation, cosine transformation)
        """
        sin_value = np.sin(2 * np.pi * value / period)
        cos_value = np.cos(2 * np.pi * value / period)
        return sin_value, cos_value
    
    def add_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic datetime features like day of week, month, etc.
        
        Args:
            df: Input dataframe with datetime column
            
        Returns:
            DataFrame with additional time-based features
        """
        result = df.copy()
        dt = pd.to_datetime(result[self.date_column])
        
        # Calendar features
        result['day_of_week'] = dt.dt.weekday
        result['day_of_month'] = dt.dt.day
        result['day_of_year'] = dt.dt.day_of_year
        result['week'] = dt.dt.isocalendar().week
        result['month'] = dt.dt.month
        result['quarter'] = dt.dt.quarter
        result['year'] = dt.dt.year
        # result["hour"] = dt.dt.hour # Activate if you have datetime data
        # result["minute"] = dt.dt.minute
        
        # Boolean flags
        result['is_leap_year'] = dt.dt.is_leap_year
        result['is_month_start'] = dt.dt.is_month_start
        result['is_month_end'] = dt.dt.is_month_end
        
        # Categorical names
        result['weekday_name'] = dt.dt.day_name()
        result['month_name'] = dt.dt.month_name()
        
        self.generated_features.extend([
            'day_of_week', 'day_of_month', 'day_of_year', 'week',
            'month', 'quarter', 'year', "hour", "minute", 
            'is_leap_year', 'is_month_start',
            'is_month_end', 'weekday_name', 'month_name'
        ])
        
        return result
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical transformations of temporal features using sine and cosine.
        
        Args:
            df: Input dataframe with basic time features already added
            
        Returns:
            DataFrame with additional cyclical features
        """
        result = df.copy()
        
        # Define cyclical periods for different features
        cyclical_features = {
            'day_of_year': 365,
            'day_of_month': 31,
            'day_of_week': 7,
            'week': 52,
            'month': 12
        }
        
        # Create cyclical features
        for feature, period in cyclical_features.items():
            if feature in df.columns:
                sin_val, cos_val = self._create_cyclical_feature(df[feature], period)
                result[f'{feature}_sin'] = sin_val
                result[f'{feature}_cos'] = cos_val
                self.generated_features.extend([f'{feature}_sin', f'{feature}_cos'])
        
        return result
    
    def get_feature_list(self) -> List[str]:
        return self.generated_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations to the dataframe.
        
        Args:
            df: Input dataframe with datetime column
            
        Returns:
            DataFrame with all engineered features
        """
        df_transformed = self.add_basic_time_features(df)
        df_transformed = self.add_cyclical_features(df_transformed)
        return df_transformed
    

class Preprocessor:
    """
    Preprocessing engine in the feature engineering/prep pipeline to output ready made features for training.
    """

    def __init__(self, input_data_path: str, inference: bool, inference_input: pl.LazyFrame):
        self.input_data_path = input_data_path
        self.inference = inference
        self.inference_input = inference_input

    def _check_if_valid(self, input_data):
        if input_data.is_empty():
            raise RuntimeError("Polars dataframe is empty. Check your input data.")
        return "valid"

    def _scan_and_wrangle(self):
        """
        Method that reads the input data but never reads into memory - this enables fast data manipulation and feature engineering.
        """
        transaction_df = pl.scan_csv(self.input_data_path, separator=",") if not self.inference else self.inference_input # Dont read into memory until needed

        # df_type = self._check_if_valid(transaction_df)
        df_type = "valid"

        if df_type in "valid":
            clean_df = transaction_df.filter(pl.col("Individual_Price_US$") != "#VALUE!").clone()
            dirty_df = transaction_df.filter(pl.col("Individual_Price_US$") == "#VALUE!").clone()

            clean_df = clean_df.with_columns([
                pl.col("Individual_Price_US$").str.replace(",", "").cast(pl.Float32).alias("Individual_Price_US$"),
                pl.col("Amount US$").str.replace(",", "").cast(pl.Float32).alias("Amount US$")
                ])
            
            fill_missing_price_df = (clean_df
                                     .group_by(["State", "Device_Type", "Category", "Product", "Gender", "Delivery_Type", "Customer_Login_type"])
                                     .agg(pl.col("Individual_Price_US$")
                                          .mean())
                                          .sort(by=["State", "Device_Type", "Category", "Product", "Gender", "Delivery_Type", "Customer_Login_type"]))
            
            dirty_df = (dirty_df
                        .join(fill_missing_price_df, 
                              how="left", 
                              on=["State", "Device_Type", "Category", "Product", "Gender", "Delivery_Type", "Customer_Login_type"])
                        .with_columns([

                            pl.col("Individual_Price_US$_right").alias("Individual_Price_US$")
                            
                            ])
                            .drop(pl.col("Individual_Price_US$_right")))
            
            dirty_df_clean = (dirty_df
                              .with_columns([
                                  (pl.col("Individual_Price_US$") * pl.col("Quantity")).alias("Amount US$").cast(pl.Float32).round(0)
                                  ]))
            
            full_clean_df = pl.concat([clean_df, dirty_df_clean], how="vertical_relaxed")

            full_clean_df = (full_clean_df
                             .with_columns([
                                 pl.col("Time").str.slice(offset=0, length=2).alias("hour").cast(pl.Int8),
                                 
                                 pl.col("Time").str.slice(offset=3, length=2).alias("minute").cast(pl.Int8),
                                 
                                 pl.col("Time").str.slice(offset=6, length=2).alias("second").cast(pl.Int8),

                                 pl.col("Date").str.to_date().alias("Date")
                                 
                                 ])
                                 .drop(pl.col("Time","Year_Month", "State_duplicated_0", "Country", "Amount US$")))
            
            return full_clean_df.collect()   # Read data into memory when finished
    
    def _inference_wrangle(self, date_feature: str):
        transaction_df = self.inference_input

        transaction_df = (transaction_df
                    .with_columns([
                        pl.col("Time").str.slice(offset=0, length=2).alias("hour").cast(pl.Int8),
                        
                        pl.col("Time").str.slice(offset=3, length=2).alias("minute").cast(pl.Int8),
                        
                        pl.col("Time").str.slice(offset=6, length=2).alias("second").cast(pl.Int8),

                        pl.col("Date").str.to_date().alias("Date")
                        
                        ]).drop(pl.col("Time"))).collect()
        
        feature_engineer = TimeSeriesFeatureEngineer( date_column=date_feature)
        engineered_df = feature_engineer.transform(transaction_df.to_pandas().copy())
        engineered_df = self._categorical_handler(engineered_df)
        return engineered_df.drop(columns="Date")

    
    def _categorical_handler(self, input_data: pd.DataFrame):
        categorical_handling_df = input_data.copy()
        string_cols = [object_col for object_col in categorical_handling_df.columns if categorical_handling_df[object_col].dtype == object]

        for col in string_cols:
            col_encoder = LabelEncoder()
            categorical_handling_df[col+"_code"] = col_encoder.fit_transform(categorical_handling_df[col].values)
        
        categorical_handling_df = categorical_handling_df.drop(columns=string_cols)
        
        return categorical_handling_df

        
    def feature_engineering(self, date_feature: str):
        feature_engineer = TimeSeriesFeatureEngineer( date_column=date_feature)

        engineered_df = self._scan_and_wrangle()
        engineered_df = feature_engineer.transform(engineered_df.to_pandas().copy())
        engineered_df = self._categorical_handler(engineered_df)

        return engineered_df



def split_training_test(input_data, test_size: float, shuffle: str):
    full_dataframe = input_data.copy()

    cols_drop = ["Transaction_Result", "Date", "Transaction_id", "customer_id"]
    X_input = full_dataframe.drop(columns=cols_drop)
    Y_input = full_dataframe["Transaction_Result"]

    X_train, X_test, y_train, y_test = train_test_split(X_input, Y_input, test_size=test_size, shuffle=shuffle)

    cols_standardize = ['Individual_Price_US$', 'Quantity']

    for col in cols_standardize:
        X_train[col] = standardize(X_train, col)
        X_test[col] = standardize(X_test, col)

    return X_train, X_test, y_train, y_test, X_input, Y_input





def standardize(input_data, target_col):
    return (input_data[target_col] - input_data[target_col].mean()) / input_data[target_col].std()

def normalize(input_data, target_col):
    return (input_data[target_col] - input_data[target_col].min()) / (input_data[target_col].max() - input_data[target_col].min())


