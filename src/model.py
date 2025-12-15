from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from .preprocessing import build_preprocessor
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier

# def build_model(preprocessor):
#     """Build a machine learning model pipeline with preprocessing and logistic regression."""
#     classifier = LogisticRegression(random_state=42, max_iter=1000)
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', classifier)
#     ])
#     return model   # Validation accuracy 87.7%

# add random forest model later

# def build_model(preprocessor):
#     """Build a machine learning model pipeline with preprocessing and random forest classifier."""
#     classifier = RandomForestClassifier(random_state=42, n_estimators=100)
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', classifier)
#     ])
#     return model   # Validation accuracy 90.17%

# Add XG Boost model later
def build_model(preprocessor):
    """Build a machine learning model pipeline with preprocessing and XGBoost classifier."""
    from xgboost import XGBClassifier
    classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return model   # Validation accuracy 87.7%



def train_model(model, X_train, y_train):
    """Train the machine learning model."""
    model.fit(X_train, y_train)
    return model


