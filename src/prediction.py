import os
import pandas as pd
from sklearn.pipeline import Pipeline
from .data_loader import load_test_data
from .config import PREDICTIONS_PATH

def generate_test_predictions(model: Pipeline, df_test: pd.DataFrame, prediction_column: str = "predicted_risk_flag",) -> str:
    """Generate predictions on the test dataset and save them to a CSV file."""
    # Generate predictions
    predictions = model.predict(df_test)
    output_df = df_test.copy()
    output_df[prediction_column] = predictions

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    output_df.to_csv(PREDICTIONS_PATH, index=False)
    return PREDICTIONS_PATH
                