"""Simple linear regression trained with gradient descent.

Designed for the AI_data learning package. Uses NumPy/Pandas/Matplotlib and
does not use scikit-learn to fit the regression model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .help import Help
except ImportError:  # Allows standalone testing, though package use is preferred.
    class Help:
        methods = {}

        def help(self, method=None):
            if method is None:
                for name, details in self.methods.items():
                    print(f"{name}: {details.get('description', '')}")
                return
            if method not in self.methods:
                raise KeyError(f"Unknown method: {method}")
            print(self.methods[method])


class SimpleLinearRegression(Help):
    """Fit y = intercept + slope*x using batch gradient descent."""

    methods = {
        "fit": {
            "description": "Train simple linear regression with batch gradient descent.",
            "parameters": "X, y, learning_rate=0.01, epochs=2000, tolerance=1e-10.",
            "returns": "Self, with learned slope_, intercept_, and loss_history_.",
            "example": "model.fit(X_train, y_train)"
        },
        "predict": {
            "description": "Predict target values for one numeric feature.",
            "parameters": "X: one-dimensional array, Series, or one-column DataFrame.",
            "returns": "NumPy array of predictions.",
            "example": "y_pred = model.predict(X_test)"
        },
        "evaluate": {
            "description": "Calculate MAE, MSE, RMSE, R-squared, and adjusted R-squared.",
            "parameters": "X, y_true.",
            "returns": "Dictionary of regression metrics.",
            "example": "metrics = model.evaluate(X_test, y_test)"
        },
        "metrics": {
            "description": "Calculate regression metrics from true and predicted values.",
            "parameters": "y_true, y_pred, n_features=1.",
            "returns": "Dictionary with MAE, MSE, RMSE, R-squared, adjusted R-squared.",
            "example": "model.metrics(y_test, y_pred)"
        },
        "residuals": {
            "description": "Return residuals (actual minus predicted).",
            "parameters": "X, y_true.",
            "returns": "NumPy array of residuals.",
            "example": "errors = model.residuals(X_test, y_test)"
        },
        "plot_regression": {
            "description": "Plot observed points and the fitted regression line.",
            "parameters": "X, y, title='Simple Linear Regression'.",
            "returns": "Displays a Matplotlib plot.",
            "example": "model.plot_regression(X_train, y_train)"
        },
        "plot_loss": {
            "description": "Plot training MSE by gradient-descent epoch.",
            "parameters": "log_y=False.",
            "returns": "Displays a Matplotlib plot.",
            "example": "model.plot_loss()"
        },
        "plot_residuals": {
            "description": "Plot residuals against predicted values.",
            "parameters": "X, y_true.",
            "returns": "Displays a residual plot.",
            "example": "model.plot_residuals(X_test, y_test)"
        }
    }

    def __init__(self):
        self.slope_ = None
        self.intercept_ = None
        self.loss_history_ = []
        self.n_features_in_ = 1
        self.n_iter_ = 0
        self._fitted = False

    @staticmethod
    def _as_1d(values, name):
        if isinstance(values, (pd.DataFrame, pd.Series)):
            array = values.to_numpy(dtype=float)
        else:
            array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            if array.shape[1] != 1:
                raise ValueError(
                    f"{name} must contain exactly one feature column; "
                    f"received {array.shape[1]} columns."
                )
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence.")
        if array.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains missing or infinite values. Clean them first.")
        return array

    def fit(self, X, y, learning_rate=0.01, epochs=2000, tolerance=1e-10):
        """Fit by gradient descent after scaling x and y internally for stability."""
        x = self._as_1d(X, "X")
        target = self._as_1d(y, "y")
        if len(x) != len(target):
            raise ValueError("X and y must have the same number of rows.")
        if len(x) < 2:
            raise ValueError("At least two observations are required.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number.")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if tolerance < 0:
            raise ValueError("tolerance cannot be negative.")

        self.x_mean_ = float(np.mean(x))
        self.x_scale_ = float(np.std(x))
        self.y_mean_ = float(np.mean(target))
        self.y_scale_ = float(np.std(target))

        if self.x_scale_ == 0:
            raise ValueError("X has no variation; a regression slope cannot be learned.")
        if self.y_scale_ == 0:
            # Constant target: best predictions are the same constant.
            self.slope_ = 0.0
            self.intercept_ = self.y_mean_
            self.loss_history_ = [0.0]
            self.n_iter_ = 0
            self._fitted = True
            return self

        xs = (x - self.x_mean_) / self.x_scale_
        ys = (target - self.y_mean_) / self.y_scale_
        m = len(xs)
        weight = 0.0
        bias = 0.0
        self.loss_history_ = []
        previous_loss = None

        for epoch in range(epochs):
            predictions = weight * xs + bias
            errors = predictions - ys
            # Gradients of mean squared error J = (1/m) * sum((prediction - y)^2).
            dw = (2.0 / m) * np.dot(xs, errors)
            db = (2.0 / m) * np.sum(errors)
            weight -= learning_rate * dw
            bias -= learning_rate * db

            current_scaled_pred = weight * xs + bias
            current_loss = float(np.mean((current_scaled_pred - ys) ** 2))
            self.loss_history_.append(current_loss)
            self.n_iter_ = epoch + 1
            if previous_loss is not None and abs(previous_loss - current_loss) <= tolerance:
                break
            previous_loss = current_loss

        # Convert learned standardized-space coefficients back to original units.
        self.slope_ = float(weight * self.y_scale_ / self.x_scale_)
        self.intercept_ = float(self.y_mean_ + self.y_scale_ * bias - self.slope_ * self.x_mean_)
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(X_train, y_train) first.")

    def predict(self, X):
        self._require_fitted()
        x = self._as_1d(X, "X")
        return self.intercept_ + self.slope_ * x

    @staticmethod
    def metrics(y_true, y_pred, n_features=1):
        """Compute common regression metrics; R² can be negative on test data."""
        actual = SimpleLinearRegression._as_1d(y_true, "y_true")
        predicted = SimpleLinearRegression._as_1d(y_pred, "y_pred")
        if len(actual) != len(predicted):
            raise ValueError("y_true and y_pred must have the same number of rows.")
        residual = actual - predicted
        mae = float(np.mean(np.abs(residual)))
        mse = float(np.mean(residual ** 2))
        rmse = float(np.sqrt(mse))
        denominator = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else float("nan")
        n = len(actual)
        adjusted_r2 = (
            float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))
            if np.isfinite(r2) and n > n_features + 1
            else float("nan")
        )
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2
        }

    def evaluate(self, X, y_true):
        self._require_fitted()
        return self.metrics(y_true, self.predict(X), n_features=1)

    def residuals(self, X, y_true):
        self._require_fitted()
        actual = self._as_1d(y_true, "y_true")
        predicted = self.predict(X)
        if len(actual) != len(predicted):
            raise ValueError("X and y_true must have the same number of rows.")
        return actual - predicted

    def plot_regression(self, X, y, title="Simple Linear Regression"):
        x = self._as_1d(X, "X")
        actual = self._as_1d(y, "y")
        if len(x) != len(actual):
            raise ValueError("X and y must have the same number of rows.")
        order = np.argsort(x)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, actual, label="Observed data")
        plt.plot(x[order], self.predict(x[order]), label="Gradient-descent line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_loss(self, log_y=False):
        self._require_fitted()
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(self.loss_history_) + 1), self.loss_history_)
        plt.xlabel("Epoch")
        plt.ylabel("Scaled training MSE")
        plt.title("Gradient Descent Convergence")
        if log_y:
            plt.yscale("log")
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, X, y_true):
        predicted = self.predict(X)
        residual = self.residuals(X, y_true)
        plt.figure(figsize=(8, 5))
        plt.scatter(predicted, residual)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted values")
        plt.ylabel("Residual (actual - predicted)")
        plt.title("Residual Plot")
        plt.tight_layout()
        plt.show()

    def equation(self, x_symbol="x"):
        self._require_fitted()
        return f"y_hat = {self.intercept_:.6g} + ({self.slope_:.6g} * {x_symbol})"
