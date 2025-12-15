from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series):
    """Evaluate the trained model on the validation set."""
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, report
