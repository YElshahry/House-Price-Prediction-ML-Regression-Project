import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy['rooms_per_household'] = X_copy['total_rooms'] / X_copy['households']
        X_copy['population_per_household'] = X_copy['population'] / X_copy['households']

        if self.add_bedrooms_per_room:
            X_copy['bedrooms_per_room'] = X_copy['total_bedrooms'] / X_copy['total_rooms']
        
        return X_copy

def preprocess_data(df):
    """
    Full preprocessing pipeline: Imputation, feature engineering, encoding.
    Returns X_train, X_test, y_train, y_test.
    """

    # Separate features and target variable
    if 'median_house_value' in df.columns:
        X = df.drop('median_house_value', axis=1)
        y = df['median_house_value']
    else:
        X = df
        y = None
    
    # Identify numerical and categorical columns
    num_attribs = list(X.select_dtypes(include=[np.number]).columns)
    cat_attribs = ['ocean_proximity']

    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_engineer', FeatureEngineer()),
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])

    # Transform data
    X_prepared = full_pipeline.fit_transform(X)

    return X_prepared, y, full_pipeline

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)