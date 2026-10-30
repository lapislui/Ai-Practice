import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .help import Help


class StandardDeviation(Help):
    """Standardize numeric data and inspect its distributions and outliers."""

    methods = {
        "train_test_split": {
            "description": "Split features and optional target into training and test sets.",
            "parameters": "X, y=None, test_size=0.3, random_state=0, shuffle=True, stratify=None.",
            "returns": "X_train, X_test; plus y_train, y_test if y is supplied.",
            "example": "X_train, X_test, y_train, y_test = sd.train_test_split(X, y)"
        },
        "fit": {
            "description": "Learn means and scales from training data.",
            "parameters": "X: numeric DataFrame or array.",
            "returns": "Self.",
            "example": "sd.fit(X_train)"
        },
        "transform": {
            "description": "Scale data with previously learned statistics.",
            "parameters": "X: same feature columns as fit data.",
            "returns": "Scaled DataFrame or NumPy array.",
            "example": "X_test_scaled = sd.transform(X_test)"
        },
        "fit_transform": {
            "description": "Fit the scaler and transform the same data.",
            "parameters": "X: numeric DataFrame or array.",
            "returns": "Scaled data.",
            "example": "X_train_scaled = sd.fit_transform(X_train)"
        },
        "standardize": {
            "description": "Standardize all numeric columns by default, or selected columns.",
            "parameters": "X=None, columns=None, fit=None. If not fitted, fits automatically; after fitting, transforms with learned statistics.",
            "returns": "Standardized DataFrame.",
            "example": "scaled_df = sd.standardize()"
        },
        "inverse_transform": {
            "description": "Restore standardized values to their original scale.",
            "parameters": "X: previously scaled data.",
            "returns": "Data in the original scale.",
            "example": "original = sd.inverse_transform(scaled_df)"
        },
        "mean": {
            "description": "Get learned means, or calculate means for all/one numeric column.",
            "parameters": "X=None, column=None.",
            "returns": "Series or scalar.",
            "example": "sd.mean() or sd.mean(column='Age')"
        },
        "variance": {
            "description": "Get learned variances, or calculate variances for all/one numeric column.",
            "parameters": "X=None, column=None, ddof=0.",
            "returns": "Series or scalar.",
            "example": "sd.variance()"
        },
        "standard_deviation": {
            "description": "Get learned scales, or calculate standard deviations for all/one numeric column.",
            "parameters": "X=None, column=None, ddof=0.",
            "returns": "Series or scalar.",
            "example": "sd.standard_deviation()"
        },
        "z_score": {
            "description": "Calculate Z-scores for all/one column or use fitted scaling statistics.",
            "parameters": "X=None, column=None, use_fitted=False.",
            "returns": "DataFrame, Series, or fitted transformed data.",
            "example": "sd.z_score(column='Age')"
        },
        "summary": {
            "description": "Show descriptive statistics for all numeric columns or one column.",
            "parameters": "X=None, column=None.",
            "returns": "DataFrame or dictionary.",
            "example": "sd.summary()"
        },
        "histogram": {
            "description": "Plot histograms for all numeric columns or one column.",
            "parameters": "X=None, column=None, bins=20, kde=True.",
            "returns": "Displays plots.",
            "example": "sd.histogram()"
        },
        "boxplot": {
            "description": "Plot boxplots for all numeric columns or one column.",
            "parameters": "X=None, column=None.",
            "returns": "Displays plots.",
            "example": "sd.boxplot(column='AnnualSalaryINR')"
        },
        "compare_distribution": {
            "description": "Compare a column before and after scaling.",
            "parameters": "original, scaled, column, kind='kde' or 'hist'.",
            "returns": "Displays side-by-side plots.",
            "example": "sd.compare_distribution(X_train, X_train_scaled, 'Age')"
        },
        "compare_scatter": {
            "description": "Compare two columns before and after scaling.",
            "parameters": "original, scaled, x, y.",
            "returns": "Displays side-by-side scatter plots.",
            "example": "sd.compare_scatter(X_train, X_train_scaled, 'Age', 'AnnualSalaryINR')"
        },
        "outlier_report": {
            "description": "Report potential outliers using the IQR rule; does not remove them.",
            "parameters": "X=None, column=None, multiplier=1.5.",
            "returns": "DataFrame of bounds and outlier values.",
            "example": "sd.outlier_report()"
        }
    }

    def __init__(self, df=None):
        self.df = df.copy() if isinstance(df, (pd.DataFrame, pd.Series)) else df
        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_names = None
        self._last_data = self.df

    @staticmethod
    def _frame(X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, pd.DataFrame):
            return X.copy()
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Input must be a 1D or 2D array, Series, or DataFrame.")
        return pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])

    @classmethod
    def _numeric_frame(cls, X):
        frame = cls._frame(X)
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("Provide at least one row and one feature column.")
        non_numeric = [
            col for col in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[col])
        ]
        if non_numeric:
            raise TypeError(
                "StandardScaler can only process numeric columns. "
                f"Non-numeric columns: {non_numeric}. Select numeric columns first."
            )
        return frame

    def _source(self, X=None):
        if X is None:
            X = self._last_data if self._last_data is not None else self.df
        if X is None:
            raise ValueError("Provide a DataFrame or initialize StandardDeviation(df).")
        return self._numeric_frame(X)

    def _select(self, X=None, column=None):
        frame = self._source(X)
        if column is not None:
            if column not in frame.columns:
                raise ValueError(f"Column '{column}' does not exist.")
            return frame, [column]
        return frame, list(frame.columns)

    def train_test_split(self, X, y=None, test_size=0.3, random_state=0,
                         shuffle=True, stratify=None):
        if stratify is not None and y is None:
            raise ValueError("Provide y when using stratify.")
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            shuffle=shuffle, stratify=stratify
        )

    def fit(self, X):
        frame = self._numeric_frame(X)
        self.scaler.fit(frame)
        self._fitted = True
        self._feature_names = list(frame.columns)
        self._last_data = frame
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted. Call sd.fit(X_train) first.")

    def _prepare(self, X):
        self._require_fitted()
        frame = self._numeric_frame(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            missing = [c for c in self._feature_names if c not in frame.columns]
            extra = [c for c in frame.columns if c not in self._feature_names]
            if missing or extra:
                raise ValueError(f"Feature mismatch. Missing: {missing}; extra: {extra}.")
            frame = frame[self._feature_names]
        elif frame.shape[1] != len(self._feature_names):
            raise ValueError(f"Expected {len(self._feature_names)} features, got {frame.shape[1]}.")
        return frame

    def _wrap(self, values, original):
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(values, columns=self._feature_names, index=original.index)
        if isinstance(original, pd.Series):
            return pd.Series(values.reshape(-1), index=original.index, name=original.name)
        return values

    def transform(self, X):
        frame = self._prepare(X)
        result = self._wrap(self.scaler.transform(frame), X)
        self._last_data = result
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        frame = self._prepare(X)
        return self._wrap(self.scaler.inverse_transform(frame), X)

    def standardize(self, X=None, columns=None, fit=None):
        """Use constructor data when X is omitted; all numeric columns by default."""
        if X is None:
            X = self.df
        if X is None:
            raise ValueError("Provide X or initialize StandardDeviation(df).")

        frame = self._numeric_frame(X)
        selected = list(frame.columns) if columns is None else (
            [columns] if isinstance(columns, str) else list(columns)
        )
        unknown = [c for c in selected if c not in frame.columns]
        if unknown:
            raise ValueError(f"Unknown columns: {unknown}")
        if not selected:
            raise ValueError("Select at least one column.")

        # Reuse an existing scaler only if its feature set matches exactly.
        if fit is True or not self._fitted:
            self.fit(frame[selected])
            return self.transform(frame[selected])

        self._require_fitted()
        if set(selected) != set(self._feature_names):
            raise ValueError(
                "Selected columns must match the columns used by fit(). "
                "Use fit=True or a separate StandardDeviation object."
            )
        return self.transform(frame[selected])

    def _stat(self, stat, X=None, column=None, ddof=0):
        if X is None and self._fitted:
            values = getattr(self.scaler, stat)
            result = pd.Series(values, index=self._feature_names, name=stat)
        else:
            frame, columns = self._select(X, column)
            if stat == "mean_":
                result = frame[columns].mean()
            elif stat == "var_":
                result = frame[columns].var(ddof=ddof)
            else:
                result = frame[columns].std(ddof=ddof)
        return result[column] if column is not None else result

    def mean(self, X=None, column=None):
        return self._stat("mean_", X, column)

    def variance(self, X=None, column=None, ddof=0):
        return self._stat("var_", X, column, ddof)

    def standard_deviation(self, X=None, column=None, ddof=0):
        # StandardScaler uses population standard deviation (ddof=0).
        return self._stat("scale_", X, column, ddof)

    def z_score(self, X=None, column=None, use_fitted=False):
        if use_fitted:
            if X is None:
                X = self.df
            if X is None:
                raise ValueError("Provide X when use_fitted=True.")
            result = self.transform(X)
        else:
            frame, columns = self._select(X, column)
            means = frame[columns].mean()
            stds = frame[columns].std(ddof=0).replace(0, np.nan)
            result = (frame[columns] - means) / stds
        if column is not None:
            if column not in result.columns:
                raise ValueError(f"Column '{column}' does not exist.")
            return result[column]
        return result

    def summary(self, X=None, column=None):
        frame, columns = self._select(X, column)
        result = frame[columns].describe().T
        return result.loc[column].to_dict() if column is not None else result

    def histogram(self, X=None, column=None, bins=20, kde=True):
        frame, columns = self._select(X, column)
        for name in columns:
            values = frame[name].dropna()
            if values.empty:
                continue
            plt.figure(figsize=(8, 5))
            sns.histplot(values, bins=bins, kde=kde and values.nunique() > 1)
            plt.title(f"Distribution: {name}")
            plt.xlabel(name)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    def boxplot(self, X=None, column=None):
        frame, columns = self._select(X, column)
        for name in columns:
            values = frame[name].dropna()
            if values.empty:
                continue
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=values)
            plt.title(f"Boxplot: {name}")
            plt.xlabel(name)
            plt.tight_layout()
            plt.show()

    def compare_distribution(self, original, scaled, column, kind="kde"):
        original = self._frame(original)
        scaled = self._frame(scaled)
        if column not in original or column not in scaled:
            raise ValueError(f"'{column}' must exist in both datasets.")
        if kind not in ("kde", "hist"):
            raise ValueError("kind must be 'kde' or 'hist'.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for frame, ax, title in (
            (original, axes[0], "Before Scaling"),
            (scaled, axes[1], "After Scaling")
        ):
            values = frame[column].dropna()
            if kind == "kde" and values.nunique() > 1:
                sns.kdeplot(values, ax=ax)
            else:
                sns.histplot(values, ax=ax)
            ax.set_title(f"{column} - {title}")
            ax.set_xlabel(column)
        plt.tight_layout()
        plt.show()

    def compare_scatter(self, original, scaled, x, y):
        original = self._frame(original)
        scaled = self._frame(scaled)
        for col in (x, y):
            if col not in original or col not in scaled:
                raise ValueError(f"'{col}' must exist in both datasets.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(original[x], original[y])
        axes[0].set(title="Before Scaling", xlabel=x, ylabel=y)
        axes[1].scatter(scaled[x], scaled[y])
        axes[1].set(title="After Scaling", xlabel=x, ylabel=y)
        plt.tight_layout()
        plt.show()

    def outlier_report(self, X=None, column=None, multiplier=1.5):
        if multiplier < 0:
            raise ValueError("multiplier must be non-negative.")
        frame, columns = self._select(X, column)
        rows = []
        for name in columns:
            values = frame[name].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            outliers = values[(values < lower) | (values > upper)]
            rows.append({
                "Column": name, "Q1": q1, "Q3": q3, "IQR": iqr,
                "Lower Bound": lower, "Upper Bound": upper,
                "Outlier Count": len(outliers), "Outlier Values": outliers.tolist()
            })
        return pd.DataFrame(rows).set_index("Column") if rows else pd.DataFrame()
