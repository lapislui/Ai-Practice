import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .AI_data.help import Help


class StandardDeviation(Help):
    """
    Standardization and Z-score analysis using sklearn StandardScaler.

    Recommended machine-learning workflow:
        sd = StandardDeviation()
        X_train, X_test, y_train, y_test = sd.train_test_split(X, y)
        sd.fit(X_train)                 # learn from training data only
        X_train_scaled = sd.transform(X_train)
        X_test_scaled = sd.transform(X_test)

    This class also supports summaries, plots, and IQR outlier reports.
    """

    methods = {
        "train_test_split": {
            "description": "Split features and an optional target into train and test sets.",
            "parameters": "X, y=None, test_size=0.3, random_state=0, shuffle=True, stratify=None.",
            "returns": "X_train, X_test; or X_train, X_test, y_train, y_test if y is supplied.",
            "example": "X_train, X_test, y_train, y_test = sd.train_test_split(X, y)"
        },
        "fit": {
            "description": "Learn each feature's mean, variance, and scale from the supplied data.",
            "parameters": "X: numeric training DataFrame or array.",
            "returns": "Self.",
            "example": "sd.fit(X_train)"
        },
        "transform": {
            "description": "Scale data using the statistics learned by fit(). Does not refit.",
            "parameters": "X: data with the same features used during fit().",
            "returns": "Scaled DataFrame or NumPy array.",
            "example": "X_test_scaled = sd.transform(X_test)"
        },
        "fit_transform": {
            "description": "Fit the scaler on data and transform the same data.",
            "parameters": "X: numeric DataFrame or array.",
            "returns": "Scaled data.",
            "example": "X_train_scaled = sd.fit_transform(X_train)"
        },
        "inverse_transform": {
            "description": "Convert standardized values back to their original scale.",
            "parameters": "X: scaled data with the fitted feature columns.",
            "returns": "Data in the original scale.",
            "example": "X_original = sd.inverse_transform(X_train_scaled)"
        },
        "mean": {
            "description": "Get learned means, or calculate means from supplied/constructor data.",
            "parameters": "X=None, column=None.",
            "returns": "One value for a column, otherwise a Series.",
            "example": "sd.mean() or sd.mean('Age')"
        },
        "variance": {
            "description": "Get learned variances, or calculate variances from supplied/constructor data.",
            "parameters": "X=None, column=None, ddof=0.",
            "returns": "One value for a column, otherwise a Series.",
            "example": "sd.variance()"
        },
        "standard_deviation": {
            "description": "Get learned scales, or calculate standard deviations from supplied/constructor data.",
            "parameters": "X=None, column=None, ddof=0.",
            "returns": "One value for a column, otherwise a Series.",
            "example": "sd.standard_deviation('Age')"
        },
        "z_score": {
            "description": "Calculate Z-scores from data, or transform with the fitted scaler.",
            "parameters": "X=None, column=None, use_fitted=False.",
            "returns": "Series for one column or DataFrame for multiple columns.",
            "example": "sd.z_score(X_train, use_fitted=True)"
        },
        "standardize": {
            "description": "Convenience wrapper: fit and scale data, or transform with existing fitted statistics.",
            "parameters": "X=None, columns=None, fit=None; fit=True refits, fit=False requires a fitted scaler.",
            "returns": "Standardized DataFrame.",
            "example": "scaled = sd.standardize(X_train, fit=True)"
        },
        "summary": {
            "description": "Return descriptive statistics for one or all numeric columns.",
            "parameters": "X=None, column=None.",
            "returns": "DataFrame, or dictionary for one column.",
            "example": "sd.summary() or sd.summary(X_train_scaled)"
        },
        "histogram": {
            "description": "Plot a histogram for one or all numeric columns.",
            "parameters": "X=None, column=None, bins=20, kde=True.",
            "returns": "Displays plots.",
            "example": "sd.histogram(X_train_scaled)"
        },
        "boxplot": {
            "description": "Plot a boxplot for one or all numeric columns.",
            "parameters": "X=None, column=None.",
            "returns": "Displays plots.",
            "example": "sd.boxplot(X_train)"
        },
        "compare_scatter": {
            "description": "Plot two numeric features before and after scaling.",
            "parameters": "original, scaled, x, y.",
            "returns": "Displays side-by-side scatter plots.",
            "example": "sd.compare_scatter(X_train, X_train_scaled, 'Age', 'EstimatedSalary')"
        },
        "compare_distribution": {
            "description": "Compare one feature's distribution before and after scaling.",
            "parameters": "original, scaled, column, kind='kde' or 'hist'.",
            "returns": "Displays side-by-side distribution plots.",
            "example": "sd.compare_distribution(X_train, X_train_scaled, 'Age')"
        },
        "outlier_report": {
            "description": "Report potential outliers using the IQR rule; does not remove them.",
            "parameters": "X=None, column=None, multiplier=1.5.",
            "returns": "DataFrame with quartiles, bounds, counts, and outlier values.",
            "example": "sd.outlier_report(X_train)"
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
        """Convert input to a DataFrame for validation and name preservation."""
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
                "StandardScaler accepts numeric columns only. Non-numeric columns: "
                + ", ".join(map(str, non_numeric))
            )
        return frame

    def _source(self, X=None):
        if X is None:
            X = self._last_data if self._last_data is not None else self.df
        if X is None:
            raise ValueError("Provide data X or initialize StandardDeviation(df).")
        return self._numeric_frame(X)

    def _selected(self, X=None, column=None):
        frame = self._source(X)
        if column is not None:
            if column not in frame.columns:
                raise ValueError(f"Column '{column}' does not exist.")
            return frame, [column]
        return frame, list(frame.columns)

    def train_test_split(
        self, X, y=None, test_size=0.3, random_state=0,
        shuffle=True, stratify=None
    ):
        """Split before fitting a scaler to avoid test-data leakage."""
        if stratify is not None and y is None:
            raise ValueError("Provide y when using stratify.")
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )

    def fit(self, X):
        """Learn StandardScaler statistics. Fit on training features only."""
        frame = self._numeric_frame(X)
        self.scaler.fit(frame)
        self._fitted = True
        self._feature_names = list(frame.columns)
        self._last_data = frame
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted. Call sd.fit(X_train) first.")

    def _prepare_transform(self, X):
        self._check_fitted()
        frame = self._numeric_frame(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            missing = [name for name in self._feature_names if name not in frame.columns]
            extra = [name for name in frame.columns if name not in self._feature_names]
            if missing or extra:
                raise ValueError(
                    f"Input columns must match fitted columns. Missing: {missing}; extra: {extra}."
                )
            frame = frame[self._feature_names]
        elif frame.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features, received {frame.shape[1]}."
            )
        return frame

    def _restore_output(self, values, original):
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(values, columns=self._feature_names, index=original.index)
        if isinstance(original, pd.Series):
            return pd.Series(values.reshape(-1), name=original.name, index=original.index)
        return values

    def transform(self, X):
        """Transform using learned training statistics; never refits."""
        frame = self._prepare_transform(X)
        values = self.scaler.transform(frame)
        self._last_data = frame
        return self._restore_output(values, X)

    def fit_transform(self, X):
        """Fit and transform the same data; usually use on training data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Reverse the fitted standardization."""
        frame = self._prepare_transform(X)
        values = self.scaler.inverse_transform(frame)
        return self._restore_output(values, X)

    def _learned_series(self, attribute, name):
        self._check_fitted()
        values = getattr(self.scaler, attribute)
        return pd.Series(values, index=self._feature_names, name=name)

    def mean(self, X=None, column=None):
        if X is None and self._fitted:
            result = self._learned_series("mean_", "mean")
        else:
            frame, columns = self._selected(X, column)
            result = frame[columns].mean()
        return result[column] if column is not None else result

    def variance(self, X=None, column=None, ddof=0):
        if X is None and self._fitted:
            result = self._learned_series("var_", "variance")
        else:
            frame, columns = self._selected(X, column)
            result = frame[columns].var(ddof=ddof)
        return result[column] if column is not None else result

    def standard_deviation(self, X=None, column=None, ddof=0):
        if X is None and self._fitted:
            result = self._learned_series("scale_", "standard_deviation")
        else:
            frame, columns = self._selected(X, column)
            result = frame[columns].std(ddof=ddof)
        return result[column] if column is not None else result

    def z_score(self, X=None, column=None, use_fitted=False):
        if use_fitted:
            if X is None:
                raise ValueError("Provide X when use_fitted=True.")
            result = self.transform(X)
            if column is not None:
                if column not in result.columns:
                    raise ValueError(f"Column '{column}' does not exist.")
                return result[column]
            return result

        frame, columns = self._selected(X, column)
        means = frame[columns].mean()
        stds = frame[columns].std(ddof=0).replace(0, np.nan)
        result = (frame[columns] - means) / stds
        return result[column] if column is not None else result

    def standardize(self, X=None, columns=None, fit=None):
        """
        Standardize selected/all columns.
        If not fitted yet, fit on X by default. If already fitted, transform with saved stats.
        Pass fit=True to explicitly fit/refit, or fit=False to require existing fitted stats.
        """
        if X is None:
            X = self.df
        if X is None:
            raise ValueError("Provide X or initialize StandardDeviation(df).")

        frame = self._numeric_frame(X)
        selected = list(frame.columns) if columns is None else (
            [columns] if isinstance(columns, str) else list(columns)
        )
        unknown = [col for col in selected if col not in frame.columns]
        if unknown:
            raise ValueError(f"Unknown columns: {unknown}")

        should_fit = (not self._fitted) if fit is None else fit
        if should_fit:
            # A scaler must be fitted to exactly the selected feature set.
            self.fit(frame[selected])
            return self.transform(frame[selected])

        self._check_fitted()
        if selected != self._feature_names:
            # Allow selecting a subset only when the scaler was fitted on that same subset.
            if set(selected) != set(self._feature_names):
                raise ValueError(
                    "The selected columns must match the columns used by fit(). "
                    "Fit a separate scaler for a different feature selection."
                )
        return self.transform(frame[selected])

    def summary(self, X=None, column=None):
        frame, columns = self._selected(X, column)
        result = frame[columns].describe().T
        if column is not None:
            return result.loc[column].to_dict()
        return result

    def histogram(self, X=None, column=None, bins=20, kde=True):
        frame, columns = self._selected(X, column)
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
        frame, columns = self._selected(X, column)
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
        original_frame = self._frame(original)
        scaled_frame = self._frame(scaled)
        for name in (x, y):
            if name not in original_frame.columns or name not in scaled_frame.columns:
                raise ValueError(f"Column '{name}' must exist in both datasets.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(original_frame[x], original_frame[y])
        axes[0].set_title("Before Scaling")
        axes[0].set_xlabel(x)
        axes[0].set_ylabel(y)
        axes[1].scatter(scaled_frame[x], scaled_frame[y], color="red")
        axes[1].set_title("After Scaling")
        axes[1].set_xlabel(x)
        axes[1].set_ylabel(y)
        plt.tight_layout()
        plt.show()

    def compare_distribution(self, original, scaled, column, kind="kde"):
        original_frame = self._frame(original)
        scaled_frame = self._frame(scaled)
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
                sns.histplot(values, ax=ax)
            ax.set_title(f"{column} — {title}")
            ax.set_xlabel(column)
        plt.tight_layout()
        plt.show()

    def outlier_report(self, X=None, column=None, multiplier=1.5):
        """Find potential IQR outliers without deleting or modifying rows."""
        if multiplier < 0:
            raise ValueError("multiplier must be non-negative.")
        frame, columns = self._selected(X, column)
        rows = []
        for name in columns:
            values = frame[name].dropna()
            if values.empty:
                continue
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            outliers = values[(values < lower) | (values > upper)]
            rows.append({
                "Column": name,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": int(len(outliers)),
                "Outlier Values": outliers.tolist()
            })
        return pd.DataFrame(rows).set_index("Column") if rows else pd.DataFrame()
