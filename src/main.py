from .data_loader import load_test_data, load_train_data
from .preprocessing import build_preprocessor, split_features_target, split_train_validation, prepare_test_fearures
from .model import build_model, train_model
from .prediction import generate_test_predictions
from .evaluation import evaluate_model

def run_pipeline():
    # Load data
    print("Loading data...")
    df_train = load_train_data()
    df_test = load_test_data()

    # Split features and target
    print("\n Preparing features and target...")
    X, y = split_features_target(df_train)

    # Build preprocessor
    print("\n Building preprocessor...")
    preprocessor = build_preprocessor(X)

    # Split into training and validation sets
    print("\n Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = split_train_validation(X, y)
    print(f"Training set size: {X_train.shape[0]} samples"
          f"\nValidation set size: {X_val.shape[0]} samples")

    # Build and train model
    print("\n Building and training model...")
    model = build_model(preprocessor)
    trained_model = train_model(model, X_train, y_train)

    # Evaluate model
    print("\n Evaluating model...")
    evaluate_model(trained_model, X_val, y_val)

    # Prepare test features
    print("\n Preparing test features...")
    X_test = prepare_test_fearures(df_test)
    print(f"Test set size: {X_test.shape[0]} samples")

    # Generate predictions on test data
    print("\n Generating test predictions...")
    predictions_path = generate_test_predictions(trained_model, X_test)
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    run_pipeline()