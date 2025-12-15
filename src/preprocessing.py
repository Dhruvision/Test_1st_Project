import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, TEST_SIZE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET_COLUMN = 'Risk_Flag' 

def split_features_target(df):
    """Split the DataFrame into features and target variable."""
    df.dropna(subset=[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def build_preprocessor(X):
    """Build a preprocessing pipeline for numerical and categorical features."""
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor

def split_train_validation(X, y):
    """Split the data into training and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_val, y_train, y_val

def prepare_test_fearures(df_test): 
    """Prepare the test features using the provided preprocessor."""
    df = df_test.copy()
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
    return df