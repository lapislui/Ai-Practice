
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .help import Help


class Univariate(Help):

    methods = {
        "value_counts": {
            "description": "Count each unique value in a column.",
            "parameters": "column: column name.",
            "returns": "Pandas Series containing value counts.",
            "example": "uni.value_counts('Sex')"
        },
        "summary": {
            "description": "Get descriptive statistics for one column.",
            "parameters": "column: column name.",
            "returns": "Pandas Series of summary statistics.",
            "example": "uni.summary('Age')"
        },
        "minimum": {
            "description": "Get the smallest numerical value.",
            "parameters": "column: numerical column name.",
            "returns": "Minimum value.",
            "example": "uni.minimum('Age')"
        },
        "maximum": {
            "description": "Get the largest numerical value.",
            "parameters": "column: numerical column name.",
            "returns": "Maximum value.",
            "example": "uni.maximum('Age')"
        },
        "mean": {
            "description": "Calculate the average of a numerical column.",
            "parameters": "column: numerical column name.",
            "returns": "Mean value.",
            "example": "uni.mean('Age')"
        },
        "median": {
            "description": "Calculate the middle value of a numerical column.",
            "parameters": "column: numerical column name.",
            "returns": "Median value.",
            "example": "uni.median('Age')"
        },
        "mode": {
            "description": "Find the most frequent value or values.",
            "parameters": "column: column name.",
            "returns": "Pandas Series of mode values.",
            "example": "uni.mode('Sex')"
        },
        "standard_deviation": {
            "description": "Measure the spread of numerical values.",
            "parameters": "column: numerical column name.",
            "returns": "Standard deviation.",
            "example": "uni.standard_deviation('Age')"
        },
        "variance": {
            "description": "Calculate the variance of numerical values.",
            "parameters": "column: numerical column name.",
            "returns": "Variance.",
            "example": "uni.variance('Age')"
        },
        "skewness": {
            "description": "Measure the asymmetry of a numerical distribution.",
            "parameters": "column: numerical column name.",
            "returns": "Skewness value.",
            "example": "uni.skewness('Age')"
        },
        "kurtosis": {
            "description": "Measure distribution tail weight relative to a normal distribution.",
            "parameters": "column: numerical column name.",
            "returns": "Kurtosis value.",
            "example": "uni.kurtosis('Age')"
        },
        "countplot": {
            "description": "Plot the frequency of categorical values.",
            "parameters": "column: column name.",
            "returns": "Matplotlib Axes.",
            "example": "uni.countplot('Embarked')"
        },
        "pie_chart": {
            "description": "Plot category proportions as a pie chart.",
            "parameters": "column: column name.",
            "returns": "Matplotlib Axes.",
            "example": "uni.pie_chart('Sex')"
        },
        "histogram": {
            "description": "Plot the distribution of numerical values.",
            "parameters": "column: numerical column name; bins=10.",
            "returns": "Matplotlib Axes.",
            "example": "uni.histogram('Age')"
        },
        "boxplot": {
            "description": "Plot spread, median, quartiles, and potential outliers.",
            "parameters": "column: numerical column name.",
            "returns": "Matplotlib Axes.",
            "example": "uni.boxplot('Age')"
        },
        "help": {
            "description": "Show all method documentation or one method's details.",
            "parameters": "method=None: optional method name.",
            "returns": "None; prints help.",
            "example": "uni.help()\nuni.help('histogram')"
        }
    }

    def __init__(self, df):
        self.df = df

    def _check_column(self, column):
        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {self.df.columns.tolist()}"
            )

    def value_counts(self, column):
        """Count occurrences of each unique value."""
        self._check_column(column)
        return self.df[column].value_counts()

    def summary(self, column):
        """Return descriptive statistics for a column."""
        self._check_column(column)

        if pd.api.types.is_numeric_dtype(self.df[column]):
            return self.df[column].describe()

        return self.df[column].describe()

    def minimum(self, column):
        """Return the minimum numerical value."""
        self._check_column(column)
        return self.df[column].min()

    def maximum(self, column):
        """Return the maximum numerical value."""
        self._check_column(column)
        return self.df[column].max()

    def mean(self, column):
        """Return the numerical average."""
        self._check_column(column)
        return self.df[column].mean()

    def median(self, column):
        """Return the numerical median."""
        self._check_column(column)
        return self.df[column].median()

    def mode(self, column):
        """Return the most frequent value or values."""
        self._check_column(column)
        return self.df[column].mode()

    def standard_deviation(self, column):
        """Return the standard deviation."""
        self._check_column(column)
        return self.df[column].std()

    def variance(self, column):
        """Return the variance."""
        self._check_column(column)
        return self.df[column].var()

    def skewness(self, column):
        """Return the skewness of a numerical column."""
        self._check_column(column)
        return self.df[column].skew()

    def kurtosis(self, column):
        """Return the kurtosis of a numerical column."""
        self._check_column(column)
        return self.df[column].kurt()

    def countplot(self, column):
        """Plot category frequencies."""
        self._check_column(column)

        ax = sns.countplot(data=self.df, x=column)
        ax.set_title(f"Count Plot of {column}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return ax

    def pie_chart(self, column):
        """Plot category proportions."""
        self._check_column(column)

        counts = self.df[column].value_counts()
        ax = counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ylabel="",
            title=f"Pie Chart of {column}"
        )
        plt.tight_layout()
        plt.show()
        return ax

    def histogram(self, column, bins=10):
        """Plot a numerical distribution."""
        self._check_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"'{column}' must be a numerical column.")

        ax = sns.histplot(
            data=self.df,
            x=column,
            bins=bins,
            kde=True
        )
        ax.set_title(f"Histogram of {column}")
        plt.tight_layout()
        plt.show()
        return ax

    def boxplot(self, column):
        """Plot a numerical distribution and potential outliers."""
        self._check_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"'{column}' must be a numerical column.")

        ax = sns.boxplot(data=self.df, x=column)
        ax.set_title(f"Box Plot of {column}")
        plt.tight_layout()
        plt.show()
        return ax
