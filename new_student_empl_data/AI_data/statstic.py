import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .help import Help


class statstic(Help):
    """
    Standardize numerical features with sklearn's StandardScaler.

    Typical workflow:
        sd = StandardDeviation()
        X_train, X_test, y_train, y_test = sd.train_test_split(X, y)
        sd.fit(X_train)
        X_train_scaled = sd.transform(X_train)
        X_test_scaled = sd.transform(X_test)

    Always split before fitting the scaler to avoid data leakage.
    """

    methods = {
        "train_test_split": {
            "description": "Split features and target into training and test sets.",
            "parameters": "X, y=None, test_size=0.3, random_state=0, shuffle=True, stratify=None.",
            "returns": "X_train, X_test if y is omitted; otherwise X_train, X_test, y_train, y_test.",
            "example": "X_train, X_test, y_train, y_test = sd.train_test_split(X, y)"
        },
        "fit": {
            "description": "Learn the training columns' means and standard deviations.",
            "parameters": "X: training DataFrame or array.",
            "returns": "Self.",
            "example": "sd.fit(X_train)"
        },
        "transform": {
            "description": "Scale data using statistics learned by fit().",
            "parameters": "X: DataFrame or array with the same feature columns.",
            "returns": "Scaled DataFrame when input is a DataFrame; otherwise a NumPy array.",
            "example": "X_test_scaled = sd.transform(X_test)"
        },
        "fit_transform": {
            "description": "Fit the scaler on data and scale that same data.",
            "parameters": "X: training DataFrame or array.",
            "returns": "Scaled data, preserving DataFrame column names when possible.",
            "example": "X_train_scaled = sd.fit_transform(X_train)"
        },
        "inverse_transform": {
            "description": "Convert scaled data back to the original feature scale.",
            "parameters": "X: scaled DataFrame or array.",
            "returns": "Data in the original scale.",
            "example": "X_original = sd.inverse_transform(X_train_scaled)"
        },
        "mean": {
            "description": "Get the means learned by the fitted scaler.",
            "parameters": "None.",
            "returns": "Pandas Series indexed by feature name, or NumPy array for array input.",
            "example": "sd.mean()"
        },
        "variance": {
            "description": "Get the variances learned by the fitted scaler.",
            "parameters": "None.",
            "returns": "Pandas Series indexed by feature name, or NumPy array for array input.",
            "example": "sd.variance()"
        },
        "standard_deviation": {
            "description": "Get the standard deviations used by the fitted scaler.",
            "parameters": "None.",
            "returns": "Pandas Series indexed by feature name, or NumPy array for array input.",
            "example": "sd.standard_deviation()"
        },
        "summary": {
            "description": "Return count, mean, standard deviation, min, quartiles, and max for numeric columns.",
            "parameters": "X: optional DataFrame; if omitted, uses the most recently stored input when available.",
            "returns": "Descriptive-statistics DataFrame.",
            "example": "sd.summary(X_train_scaled)"
        },
        "standardize": {
            "description": "Convenience method to fit and scale data, or transform with an already-fitted scaler.",
            "parameters": "X: DataFrame or array; fit=True fits on X, fit=False uses existing fitted statistics.",
            "returns": "Scaled DataFrame when input is a DataFrame; otherwise a NumPy array.",
            "example": "scaled_df = sd.standardize(X_train, fit=True)"
        },
        "histogram": {
            "description": "Plot histograms for one or all numeric columns.",
            "parameters": "X, column=None, bins=20, kde=True.",
            "returns": "Displays plots.",
            "example": "sd.histogram(X_train_scaled)"
        },
        "boxplot": {
            "description": "Plot boxplots for one or all numeric columns.",
            "parameters": "X, column=None.",
            "returns": "Displays plots.",
            "example": "sd.boxplot(X_train)"
        },
        "compare_scatter": {
            "description": "Plot two numeric features before and after standardization.",
            "parameters": "original, scaled, x, y.",
            "returns": "Displays side-by-side scatter plots.",
            "example": "sd.compare_scatter(X_train, X_train_scaled, 'Age', 'EstimatedSalary')"
        },
        "compare_distribution": {
            "description": "Compare one feature's distribution before and after standardization.",
            "parameters": "original, scaled, column, kind='kde' or 'hist'.",
            "returns": "Displays side-by-side distribution plots.",
            "example": "sd.compare_distribution(X_train, X_train_scaled, 'Age')"
        },
        "outlier_report": {
            "description": "Report potential outliers using the IQR rule; does not remove them.",
            "parameters": "X, multiplier=1.5.",
            "returns": "DataFrame with quartiles, bounds, and outlier counts.",
            "example": "sd.outlier_report(X_train)"
        }
    }

    def __init__(self, df=None):
        self.df = df.copy() if isinstance(df, pd.DataFrame) else df
        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_names = None
        self._last_data = self.df

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, pd.DataFrame):
            return X.copy()
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("X must be a 1D or 2D array, Series, or DataFrame.")
        return pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])

    @staticmethod
    def _check_numeric(X):
        frame = StandardDeviation._to_frame(X)
        non_numeric = [
            col for col in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[col])
        ]
        if non_numeric:
            raise TypeError(
                "StandardScaler requires numeric features. Non-numeric columns: "
                + ", ".join(map(str, non_numeric))
            )
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("Provide at least one row and one feature column.")
        return frame

    def _source(self, X=None):
        if X is None:
            X = self._last_data if self._last_data is not None else self.df
        if X is None:
            raise ValueError("Provide X or initialize StandardDeviation(df).")
        return self._check_numeric(X)

    def train_test_split(
        self, X, y=None, test_size=0.3, random_state=0,
        shuffle=True, stratify=None
    ):
        """Use sklearn's train_test_split; split before fitting the scaler."""
        if stratify is not None and y is None:
            raise ValueError("stratify requires a target y.")
        result = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )
        return result

    def fit(self, X):
        """Fit StandardScaler on training features only."""
        frame = self._check_numeric(X)
        self.scaler.fit(frame)
        self._fitted = True
        self._feature_names = list(frame.columns)
        self._last_data = frame
        return self

    def _check_fitted_columns(self, X):
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted. Call sd.fit(X_train) first.")
        frame = self._check_numeric(X)

        if isinstance(X, pd.DataFrame):
            missing = [col for col in self._feature_names if col not in frame.columns]
            extra = [col for col in frame.columns if col not in self._feature_names]
            if missing or extra:
                raise ValueError(
                    f"Columns do not match fitted data. Missing: {missing}; extra: {extra}."
                )
            frame = frame[self._feature_names]
        elif frame.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features, got {frame.shape[1]}."
            )
        return frame

    def _wrap_result(self, values, original):
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(values, columns=self._feature_names, index=original.index)
        if isinstance(original, pd.Series):
            return pd.Series(values.reshape(-1), name=original.name, index=original.index)
        return values

    def transform(self, X):
        """Transform using fitted training statistics; does not refit."""
        frame = self._check_fitted_columns(X)
        values = self.scaler.transform(frame)
        self._last_data = frame
        return self._wrap_result(values, X)

    def fit_transform(self, X):
        """Fit and transform the supplied data. Usually use this on training data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Reverse scaling using the fitted scaler."""
        frame = self._check_fitted_columns(X)
        values = self.scaler.inverse_transform(frame)
        return self._wrap_result(values, X)

    def mean(self):
        if not self._fitted:
            raise RuntimeError("Call fit(X_train) before requesting learned means.")
        return pd.Series(self.scaler.mean_, index=self._feature_names, name="mean")

    def variance(self):
        if not self._fitted:
            raise RuntimeError("Call fit(X_train) before requesting learned variances.")
        return pd.Series(self.scaler.var_, index=self._feature_names, name="variance")

    def standard_deviation(self):
        if not self._fitted:
            raise RuntimeError("Call fit(X_train) before requesting learned standard deviations.")
        return pd.Series(self.scaler.scale_, index=self._feature_names, name="standard_deviation")

    def standardize(self, X=None, fit=False):
        """
        If fit=True, fit and transform X. Otherwise, transform using stored statistics.
        For a first-time scaling operation, call fit=True on the training data.
        """
        if X is None:
            X = self.df
        if X is None:
            raise ValueError("Provide X or initialize StandardDeviation(df).")
        if fit:
            return self.fit_transform(X)
        return self.transform(X)

    def summary(self, X=None):
        if X is None:
            X = self._last_data if self._last_data is not None else self.df
        if X is None:
            raise ValueError("Provide X or initialize StandardDeviation(df).")
        frame = self._to_frame(X)
        return frame.describe().T

    def histogram(self, X=None, column=None, bins=20, kde=True):
        frame = self._source(X)
        columns = [column] if column is not None else list(frame.columns)
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Unknown columns: {missing}")
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
        frame = self._source(X)
        columns = [column] if column is not None else list(frame.columns)
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Unknown columns: {missing}")
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

    def compare_scatter(self, original, scaled, x, y):
        original_frame = self._to_frame(original)
        scaled_frame = self._to_frame(scaled)
        for name in (x, y):
            if name not in original_frame.columns or name not in scaled_frame.columns:
                raise ValueError(f"Column '{name}' must exist in both datasets.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(original_frame[x], original_frame[y])
        axes[0].set_title("Before Scaling")
        axes[0].set_xlabel(x)
        axes[0].set_ylabel(y)
        axes[1].scatter(scaled_frame[x], scaled_frame[y])
        axes[1].set_title("After Scaling")
        axes[1].set_xlabel(x)
        axes[1].set_ylabel(y)
        plt.tight_layout()
        plt.show()

    def compare_distribution(self, original, scaled, column, kind="kde"):
        original_frame = self._to_frame(original)
        scaled_frame = self._to_frame(scaled)
        if column not in original_frame.columns or column not in scaled_frame.columns:
            raise ValueError(f"Column '{column}' must exist in both datasets.")
        if kind not in ("kde", "hist"):
            raise ValueError("kind must be 'kde' or 'hist'.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for frame, ax, title in (
            (original_frame, axes[0], "Before Scaling"),
            (scaled_frame, axes[1], "After Scaling")
        ):
            values = frame[column].dropna()
            if kind == "kde" and values.nunique() > 1:
                sns.kdeplot(values, ax=ax)
            else:
                sns.histplot(values, kde=False, ax=ax)
            ax.set_title(f"{column} — {title}")
            ax.set_xlabel(column)
        plt.tight_layout()
        plt.show()

    def outlier_report(self, X=None, multiplier=1.5):
        """Report potential outliers using the interquartile range (IQR) rule."""
        frame = self._source(X)
        rows = []
        for column in frame.columns:
            values = frame[column].dropna()
            if values.empty:
                continue
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            outliers = values[(values < lower) | (values > upper)]
            rows.append({
                "Column": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": int(len(outliers)),
                "Outlier Values": outliers.tolist()
            })
        return pd.DataFrame(rows).set_index("Column") if rows else pd.DataFrame()
