"""Example: use AI_data to train and evaluate simple linear regression.

Run this script from the project directory containing:
    AI_data/
    student_ml_linked.csv

Required package export:
    from .simple_linear_regression import SimpleLinearRegression
in AI_data/__init__.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import AI_data as aid
from AI_data.SLR import SimpleLinearRegression


DATA_FILE = "student_ml_linked.csv"
FEATURE = "study_hours_per_day"
TARGET = "final_exam_score"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    # Use AI_data's existing data loader.
    loader = aid.LoadData(DATA_FILE)
    df = loader.read()

    print("\n=== Dataset inspection ===")
    definition = aid.DataDefinition(df)
    print("Shape:", definition.shape())
    print("Columns:", definition.columns())
    print("\nMissing values:\n", definition.missing_values())

    required = [FEATURE, TARGET]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Required columns not found: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the two columns for this first simple-regression experiment.
    # Convert invalid/missing values to NaN and remove those rows for this exercise.
    data = df[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if len(data) < 5:
        raise ValueError("Not enough valid rows to train and evaluate the model.")

    X = data[[FEATURE]]
    y = data[TARGET]

    # Use the existing StandardDeviation helper's train/test split method if available.
    # Splitting is done BEFORE any fitting to keep the test set unseen.
    if hasattr(aid, "StandardDeviation"):
        splitter = aid.StandardDeviation(df)
        X_train, X_test, y_train, y_test = splitter.train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

    # Train our AI_data model. The model internally scales x/y for stable gradient descent,
    # then reports slope and intercept in the original units.
    model = SimpleLinearRegression()
    model.fit(
        X_train,
        y_train,
        learning_rate=0.01,
        epochs=5000,
        tolerance=1e-12
    )

    print("\n=== Learned model ===")
    print("Equation:", model.equation(TARGET))
    print(f"Slope:     {model.slope_:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"Epochs:    {model.n_iter_}")

    # Evaluate on held-out test data.
    metrics = model.evaluate(X_test, y_test)
    print("\n=== Test-set regression metrics ===")
    for name, value in metrics.items():
        print(f"{name:>12}: {value:.6f}")

    # Compare actual and predicted values for a few test rows.
    predictions = model.predict(X_test)
    comparison = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions,
        "Residual": y_test.to_numpy() - predictions
    }, index=y_test.index)
    print("\n=== Actual vs predicted (first 10 test rows) ===")
    print(comparison.head(10).round(3).to_string())

    # Optional relationship exploration with the existing Bivariate class.
    if hasattr(aid, "Bivariate"):
        try:
            print("\n=== Correlation ===")
            print(aid.Bivariate(data).correlation())
        except (AttributeError, TypeError, ValueError) as exc:
            print("Skipping package correlation helper:", exc)

    # Visual diagnostics.
    model.plot_regression(X_train, y_train, title="Study Hours vs Exam Score (Training Set)")
    model.plot_loss()
    model.plot_residuals(X_test, y_test)

    # Save test predictions beside the script for later inspection.
    output_path = Path("simple_linear_regression_test_predictions.csv")
    comparison.to_csv(output_path, index=True)
    print(f"\nSaved test predictions to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
