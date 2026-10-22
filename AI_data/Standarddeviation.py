import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .help import Help


class StandardDeviation(Help):
    methods = {
        "mean": {
            "description": "Calculate the mean of one or all numerical columns.",
            "parameters": "column=None: optional numerical column name.",
            "returns": "Number for one column, Series for all numerical columns.",
            "example": "sd.mean() or sd.mean('Age')"
        },
        "variance": {
            "description": "Calculate sample/population variance of one or all numerical columns.",
            "parameters": "column=None: optional column name; population=False.",
            "returns": "Number for one column, Series for all numerical columns.",
            "example": "sd.variance() or sd.variance('Age')"
        },
        "standard_deviation": {
            "description": "Calculate standard deviation of one or all numerical columns.",
            "parameters": "column=None: optional column name; population=False.",
            "returns": "Number for one column, Series for all numerical columns.",
            "example": "sd.standard_deviation() or sd.standard_deviation('Age')"
        },
        "z_score": {
            "description": "Calculate Z-scores for one or all numerical columns.",
            "parameters": "column=None: optional numerical column name; population=False.",
            "returns": "Series for one column or DataFrame for all numerical columns.",
            "example": "sd.z_score() or sd.z_score('Age')"
        },
        "standardize": {
            "description": "Standardize selected or all numerical columns manually.",
            "parameters": "columns=None: optional column name or list; population=False.",
            "returns": "Copy of the DataFrame with selected columns standardized.",
            "example": "sd.standardize() or sd.standardize(['Age', 'AnnualSalaryINR'])"
        },
        "summary": {
            "description": "Show statistics for one or all numerical columns.",
            "parameters": "column=None: optional numerical column name; population=False.",
            "returns": "Dictionary for one column or DataFrame for all numerical columns.",
            "example": "sd.summary() or sd.summary('Age')"
        },
        "histogram": {
            "description": "Plot histograms for one or all numerical columns.",
            "parameters": "column=None: optional column name; bins=10.",
            "returns": "Displays one or more histograms.",
            "example": "sd.histogram() or sd.histogram('Age')"
        },
        "boxplot": {
            "description": "Plot boxplots for one or all numerical columns.",
            "parameters": "column=None: optional column name.",
            "returns": "Displays one or more boxplots.",
            "example": "sd.boxplot() or sd.boxplot('Age')"
        }
    }

    def __init__(self, df):
        self.df = df

    def _numeric_columns(self):
        columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not columns:
            raise ValueError("No numerical columns were found in the DataFrame.")
        return columns

    def _check_column(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"Column '{column}' is not numerical.")

    def _columns(self, column=None):
        if column is None:
            return self._numeric_columns()
        self._check_column(column)
        return [column]

    def mean(self, column=None):
        result = self.df[self._columns(column)].mean()
        return result[column] if column is not None else result

    def variance(self, column=None, population=False):
        result = self.df[self._columns(column)].var(ddof=0 if population else 1)
        return result[column] if column is not None else result

    def standard_deviation(self, column=None, population=False):
        result = self.df[self._columns(column)].std(ddof=0 if population else 1)
        return result[column] if column is not None else result

    def z_score(self, column=None, population=False):
        columns = self._columns(column)
        result = pd.DataFrame(index=self.df.index)
        ddof = 0 if population else 1

        for name in columns:
            values = self.df[name]
            mean = values.mean()
            std = values.std(ddof=ddof)
            result[name] = (values - mean) / std if pd.notna(std) and std != 0 else np.nan

        return result[column] if column is not None else result

    def standardize(self, columns=None, population=False):
        if columns is None:
            columns = self._numeric_columns()
        elif isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)

        result = self.df.copy()
        ddof = 0 if population else 1

        for name in columns:
            self._check_column(name)
            values = self.df[name]
            std = values.std(ddof=ddof)
            if pd.isna(std) or std == 0:
                raise ValueError(f"Cannot standardize '{name}': standard deviation is zero or undefined.")
            result[name] = (values - values.mean()) / std

        return result

    def summary(self, column=None, population=False):
        columns = self._columns(column)
        selected = self.df[columns]
        ddof = 0 if population else 1

        result = pd.DataFrame({
            "Count": selected.count(),
            "Mean": selected.mean(),
            "Variance": selected.var(ddof=ddof),
            "Standard Deviation": selected.std(ddof=ddof),
            "Minimum": selected.min(),
            "Median": selected.median(),
            "Maximum": selected.max()
        })

        return result.loc[column].to_dict() if column is not None else result

    def histogram(self, column=None, bins=10):
        for name in self._columns(column):
            values = self.df[name].dropna()
            if values.empty:
                continue

            mean = values.mean()
            std = values.std()
            plt.figure(figsize=(8, 5))
            sns.histplot(values, bins=bins, kde=values.nunique() > 1)
            plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")

            if pd.notna(std):
                plt.axvline(mean + std, color="green", linestyle="--",
                            label=f"Mean + SD: {mean + std:.2f}")
                plt.axvline(mean - std, color="green", linestyle="--",
                            label=f"Mean - SD: {mean - std:.2f}")

            plt.title(f"Distribution and Standard Deviation: {name}")
            plt.xlabel(name)
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def boxplot(self, column=None):
        for name in self._columns(column):
            values = self.df[name].dropna()
            if values.empty:
                continue

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=values)
            plt.title(f"Boxplot of {name}")
            plt.xlabel(name)
            plt.tight_layout()
            plt.show()
