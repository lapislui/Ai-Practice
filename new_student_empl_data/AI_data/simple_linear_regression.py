"""
Simple Linear Regression for AI_data.

Designed for the same lightweight workflow as:
    import AI_data as aid
    loader = aid.LoadData("employee_data_practice.csv")
    df = loader.read()
    reg = aid.SimpleLinearRegression(df)
    reg.fit("experience_years", "salary")
    reg.summary()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class SimpleLinearRegression:
    """A small, beginner-friendly simple linear regression helper."""

    def __init__(self, data):
        self.data = data.copy()
        self.model = None
        self.x_column = None
        self.y_column = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.history = []
        self._fitted = False

    def help(self, method=None):
        methods = {
            "fit": "fit(x, y, test_size=0.2, random_state=42, epochs=1000) - train the model.",
            "predict": "predict(values) - predict target values for one or more x values.",
            "summary": "summary() - show equation and evaluation metrics.",
            "evaluate": "evaluate() - return MAE, MSE, RMSE, R2 and adjusted R2.",
            "plot": "plot() - display the data points and fitted regression line.",
            "loss_plot": "loss_plot() - display training loss by epoch.",
            "residual_plot": "residual_plot() - display residuals.",
        }
        if method:
            print(methods.get(method, f"No help available for '{method}'."))
        else:
            print("SimpleLinearRegression methods:")
            for name, description in methods.items():
                print(f"  {description}")

    def fit(self, x, y, test_size=0.2, random_state=42, epochs=1000):
        """Train with sklearn SGDRegressor (stochastic gradient descent)."""
        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError("Please provide valid column names from your DataFrame.")

        clean = self.data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(clean) < 3:
            raise ValueError("At least 3 valid rows are required.")
        if clean[x].nunique() < 2:
            raise ValueError(f"'{x}' must contain at least two different values.")

        X = clean[[x]].to_numpy()
        Y = clean[[y]].to_numpy()
        self.x_column, self.y_column = x, y

        # Hold out a test set for honest evaluation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        self.x_scaler.fit(X_train)
        self.y_scaler.fit(y_train)
        X_train_s = self.x_scaler.transform(X_train)
        X_test_s = self.x_scaler.transform(X_test)
        y_train_s = self.y_scaler.transform(y_train).ravel()

        self.model = SGDRegressor(
            loss="squared_error", penalty=None, learning_rate="constant",
            eta0=0.01, random_state=random_state
        )
        self.history = []
        for _ in range(int(epochs)):
            self.model.partial_fit(X_train_s, y_train_s)
            pred_s = self.model.predict(X_train_s)
            self.history.append(float(np.mean((y_train_s - pred_s) ** 2)))

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train.ravel(), y_test.ravel()
        self.train_predictions = self._predict_array(X_train)
        self.test_predictions = self._predict_array(X_test)
        self._fitted = True
        return self

    def _predict_array(self, values):
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        scaled = self.x_scaler.transform(values)
        predicted_scaled = self.model.predict(scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(predicted_scaled).ravel()

    def predict(self, values):
        """Predict y for a number or a list/array of x values."""
        self._check_fitted()
        scalar = np.isscalar(values)
        result = self._predict_array([values] if scalar else values)
        return float(result[0]) if scalar else result

    def evaluate(self):
        """Return metrics for the held-out test set."""
        self._check_fitted()
        actual, predicted = self.y_test, self.test_predictions
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted) if len(actual) > 1 else float("nan")
        n, p = len(actual), 1
        adjusted = (1 - (1-r2)*(n-1)/(n-p-1)) if n > p+1 else float("nan")
        return {
            "MAE": float(mae), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R2": float(r2),
            "Adjusted R2": float(adjusted)
        }

    def summary(self):
        """Print the fitted equation and test metrics."""
        self._check_fitted()
        slope = float(self.model.coef_[0] * self.y_scaler.scale_[0] / self.x_scaler.scale_[0])
        intercept = float(self.y_scaler.mean_[0] +
                          self.y_scaler.scale_[0] * self.model.intercept_[0] -
                          slope * self.x_scaler.mean_[0])
        print("Simple Linear Regression")
        print(f"X: {self.x_column} | Y: {self.y_column}")
        print(f"Equation: {self.y_column} = {intercept:.4f} + ({slope:.4f} * {self.x_column})")
        print("\nTest metrics:")
        for key, value in self.evaluate().items():
            print(f"{key}: {value:.4f}")
        return self.evaluate()

    def plot(self):
        """Plot original observations and the fitted line."""
        self._check_fitted()
        x = pd.to_numeric(self.data[self.x_column], errors="coerce")
        y = pd.to_numeric(self.data[self.y_column], errors="coerce")
        valid = x.notna() & y.notna()
        order = np.argsort(x[valid].to_numpy())
        xs = x[valid].to_numpy()[order]
        plt.scatter(x[valid], y[valid], color="steelblue", label="Actual data")
        plt.plot(xs, self.predict(xs), color="crimson", label="Regression line")
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.title("Simple Linear Regression")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def loss_plot(self):
        self._check_fitted()
        plt.plot(self.history)
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error (scaled)")
        plt.title("Gradient Descent Training Loss")
        plt.grid(True, alpha=0.3)
        plt.show()

    def residual_plot(self):
        self._check_fitted()
        residuals = self.y_test - self.test_predictions
        plt.scatter(self.test_predictions, residuals)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.grid(True, alpha=0.3)
        plt.show()

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit(x, y) before using this method.")
