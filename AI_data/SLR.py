"""Simple linear regression using scikit-learn's SGDRegressor.

This implementation uses stochastic gradient descent (SGD) to learn a one-feature
linear model and scikit-learn metrics to evaluate predictions.

Public API:
    SimpleLinearRegression.fit(X, y, ...)
    SimpleLinearRegression.predict(X)
    SimpleLinearRegression.evaluate(X, y_true)
    SimpleLinearRegression.metrics(y_true, y_pred)
    SimpleLinearRegression.residuals(X, y_true)
    SimpleLinearRegression.plot_regression(X, y)
    SimpleLinearRegression.plot_loss()
    SimpleLinearRegression.plot_residuals(X, y_true)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from .help import Help
except ImportError:
    # Fallback keeps the module importable if the shared Help class is unavailable.
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
    """Fit y = intercept + slope*x using scikit-learn SGDRegressor."""

    methods = {
        "fit": {
            "description": "Train a one-feature linear model using stochastic gradient descent.",
            "parameters": "X, y, learning_rate=0.01, epochs=2000, tolerance=1e-6, random_state=42.",
            "returns": "Self, with slope_, intercept_, loss_history_, and sklearn_model_.",
            "example": "model.fit(X_train, y_train)"
        },
        "predict": {
            "description": "Predict target values for one numeric feature.",
            "parameters": "X: one-dimensional array, Series, or one-column DataFrame.",
            "returns": "NumPy array of predictions.",
            "example": "y_pred = model.predict(X_test)"
        },
        "evaluate": {
            "description": "Calculate MAE, MSE, RMSE, R-squared, and adjusted R-squared with sklearn.metrics.",
            "parameters": "X, y_true.",
            "returns": "Dictionary of regression metrics.",
            "example": "metrics = model.evaluate(X_test, y_test)"
        },
        "metrics": {
            "description": "Calculate regression metrics from true and predicted values using sklearn.metrics.",
            "parameters": "y_true, y_pred, n_features=1.",
            "returns": "Dictionary containing MAE, MSE, RMSE, R2, and Adjusted_R2.",
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
            "description": "Plot the SGD training loss across epochs.",
            "parameters": "No parameters.",
            "returns": "Displays a loss plot using loss_curve_ from SGDRegressor.",
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
        self.sklearn_model_ = None
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
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

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        epochs=2000,
        tolerance=1e-6,
        random_state=42,
    ):
        """Fit using sklearn SGDRegressor after standardizing X and y.

        `learning_rate` is the SGD step size (`eta0`) when using the constant
        learning-rate schedule. X and y are standardized for numerical stability;
        predictions, slope_, and intercept_ are returned in original units.
        """
        x = self._as_1d(X, "X").reshape(-1, 1)
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

        if np.std(x) == 0:
            raise ValueError("X has no variation; a regression slope cannot be learned.")

        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        x_scaled = self.x_scaler_.fit_transform(x)
        y_scaled = self.y_scaler_.fit_transform(target.reshape(-1, 1)).ravel()

        self.sklearn_model_ = SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="constant",
            eta0=learning_rate,
            max_iter=1,
            tol=None,
            random_state=random_state,
            shuffle=True,
            fit_intercept=True,
            average=False,
            warm_start=True,
        )
        self.loss_history_ = []
        previous_loss = None
        self.n_iter_ = 0
        # partial_fit performs one SGD training pass each time. Keeping the loop
        # here lets us record a comparable standardized-target MSE each epoch.
        for epoch in range(epochs):
            self.sklearn_model_.partial_fit(x_scaled, y_scaled)
            epoch_predictions = self.sklearn_model_.predict(x_scaled)
            current_loss = float(mean_squared_error(y_scaled, epoch_predictions))
            self.loss_history_.append(current_loss)
            self.n_iter_ = epoch + 1
            if previous_loss is not None and abs(previous_loss - current_loss) <= tolerance:
                break
            previous_loss = current_loss

        # Convert standardized-space coefficient back to original data units.
        scaled_slope = float(self.sklearn_model_.coef_[0])
        self.slope_ = float(
            scaled_slope * self.y_scaler_.scale_[0] / self.x_scaler_.scale_[0]
        )
        self.intercept_ = float(
            self.y_scaler_.mean_[0]
            + self.y_scaler_.scale_[0] * self.sklearn_model_.intercept_[0]
            - self.slope_ * self.x_scaler_.mean_[0]
        )
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(X_train, y_train) first.")

    def predict(self, X):
        self._require_fitted()
        x = self._as_1d(X, "X").reshape(-1, 1)
        x_scaled = self.x_scaler_.transform(x)
        y_scaled_pred = self.sklearn_model_.predict(x_scaled).reshape(-1, 1)
        return self.y_scaler_.inverse_transform(y_scaled_pred).ravel()

    @staticmethod
    def metrics(y_true, y_pred, n_features=1):
        """Compute regression metrics with sklearn.metrics."""
        actual = SimpleLinearRegression._as_1d(y_true, "y_true")
        predicted = SimpleLinearRegression._as_1d(y_pred, "y_pred")
        if len(actual) != len(predicted):
            raise ValueError("y_true and y_pred must have the same number of rows.")

        mae = float(mean_absolute_error(actual, predicted))
        mse = float(mean_squared_error(actual, predicted))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan")
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
            "Adjusted_R2": adjusted_r2,
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

    def equation(self, x_symbol="x"):
        self._require_fitted()
        return f"y_hat = {self.intercept_:.6g} + ({self.slope_:.6g} * {x_symbol})"

    def plot_regression(self, X, y, title="Simple Linear Regression"):
        x = self._as_1d(X, "X")
        actual = self._as_1d(y, "y")
        if len(x) != len(actual):
            raise ValueError("X and y must have the same number of rows.")
        order = np.argsort(x)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, actual, label="Observed data")
        plt.plot(x[order], self.predict(x[order]), label="scikit-learn SGD regression line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        self._require_fitted()
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(self.loss_history_) + 1), self.loss_history_)
        plt.xlabel("Epoch")
        plt.ylabel("Training loss (standardized target)")
        plt.title("scikit-learn SGDRegressor Loss Curve")
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
