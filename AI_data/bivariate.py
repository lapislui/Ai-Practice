
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .help import Help


class Bivariate(Help):

    methods = {
        "scatterplot": {
            "description": "Show the relationship between two numerical columns.",
            "parameters": "x, y: numerical column names; hue=None: optional category.",
            "returns": "Matplotlib Axes.",
            "example": "bi.scatterplot('total_bill', 'tip', hue='sex')"
        },
        "barplot": {
            "description": "Compare a numerical value across categories using an aggregate, usually the mean.",
            "parameters": "x: categorical column; y: numerical column; hue=None; estimator='mean'.",
            "returns": "Matplotlib Axes.",
            "example": "bi.barplot('Pclass', 'Age', hue='Sex')"
        },
        "boxplot": {
            "description": "Compare numerical distributions across categories.",
            "parameters": "x: categorical column; y: numerical column; hue=None.",
            "returns": "Matplotlib Axes.",
            "example": "bi.boxplot('Sex', 'Age', hue='Survived')"
        },
        "kdeplot": {
            "description": "Compare smoothed numerical distributions, optionally by category.",
            "parameters": "x: numerical column; hue=None: category column.",
            "returns": "Matplotlib Axes.",
            "example": "bi.kdeplot('Age', hue='Survived')"
        },
        "crosstab": {
            "description": "Create a frequency table for two categorical columns.",
            "parameters": "x, y: column names; normalize=False.",
            "returns": "Pandas DataFrame.",
            "example": "bi.crosstab('Pclass', 'Survived')"
        },
        "heatmap": {
            "description": "Visualize the frequency table of two categorical columns.",
            "parameters": "x, y: categorical columns; annot=True.",
            "returns": "Matplotlib Axes.",
            "example": "bi.heatmap('Pclass', 'Survived')"
        },
        "clustermap": {
            "description": "Cluster and visualize patterns in a categorical frequency table.",
            "parameters": "x, y: categorical columns; method='average'.",
            "returns": "Seaborn ClusterGrid.",
            "example": "bi.clustermap('Parch', 'Survived')"
        },
        "pairplot": {
            "description": "Plot pairwise relationships between multiple numerical columns.",
            "parameters": "columns=None: selected numerical columns; hue=None.",
            "returns": "Seaborn PairGrid.",
            "example": "bi.pairplot(hue='species')"
        },
        "lineplot": {
            "description": "Show how a numerical value changes across an ordered or numerical variable.",
            "parameters": "x, y: column names; hue=None.",
            "returns": "Matplotlib Axes.",
            "example": "bi.lineplot('year', 'passengers')"
        },
        "pivot_table": {
            "description": "Summarize a numerical column across row and column categories.",
            "parameters": "values, index, columns: column names; aggfunc='mean'.",
            "returns": "Pandas DataFrame.",
            "example": "bi.pivot_table('passengers', 'month', 'year')"
        },
        "correlation": {
            "description": "Calculate correlations between numerical columns.",
            "parameters": "columns=None: optional list of numerical columns.",
            "returns": "Pandas DataFrame.",
            "example": "bi.correlation()"
        },
        "group_summary": {
            "description": "Summarize a numerical column for each category.",
            "parameters": "category, value: column names; aggfunc='mean'.",
            "returns": "Pandas Series or DataFrame.",
            "example": "bi.group_summary('Sex', 'Age')"
        },
        "help": {
            "description": "Show all documented methods or details for one method.",
            "parameters": "method=None: optional method name.",
            "returns": "None; prints documentation.",
            "example": "bi.help()\nbi.help('scatterplot')"
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

    def _check_numeric(self, column):
        self._check_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(
                f"'{column}' must be a numerical column."
            )

    def scatterplot(self, x, y, hue=None):
        """Plot the relationship between two numerical columns."""
        self._check_numeric(x)
        self._check_numeric(y)

        if hue is not None:
            self._check_column(hue)

        ax = sns.scatterplot(
            data=self.df, x=x, y=y, hue=hue
        )
        ax.set_title(f"{x} vs {y}")
        plt.tight_layout()
        plt.show()
        return ax

    def barplot(self, x, y, hue=None, estimator="mean"):
        """Compare a numerical value across categories."""
        self._check_column(x)
        self._check_numeric(y)

        if hue is not None:
            self._check_column(hue)

        if estimator == "mean":
            func = "mean"
        elif estimator == "sum":
            func = "sum"
        elif estimator == "median":
            func = "median"
        else:
            raise ValueError(
                "estimator must be 'mean', 'sum', or 'median'."
            )

        ax = sns.barplot(
            data=self.df, x=x, y=y, hue=hue, estimator=func
        )
        ax.set_title(f"{y} by {x}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return ax

    def boxplot(self, x, y, hue=None):
        """Compare numerical distributions across categories."""
        self._check_column(x)
        self._check_numeric(y)

        if hue is not None:
            self._check_column(hue)

        ax = sns.boxplot(
            data=self.df, x=x, y=y, hue=hue
        )
        ax.set_title(f"{y} by {x}")
        plt.tight_layout()
        plt.show()
        return ax

    def kdeplot(self, x, hue=None):
        """Plot the distribution of a numerical column."""
        self._check_numeric(x)

        if hue is not None:
            self._check_column(hue)

        ax = sns.kdeplot(
            data=self.df, x=x, hue=hue
        )
        ax.set_title(f"Distribution of {x}")
        plt.tight_layout()
        plt.show()
        return ax

    def crosstab(self, x, y, normalize=False):
        """Create a frequency table for two columns."""
        self._check_column(x)
        self._check_column(y)

        return pd.crosstab(
            self.df[x],
            self.df[y],
            normalize=normalize
        )

    def heatmap(self, x, y, annot=True):
        """Plot a heatmap of category frequencies."""
        table = self.crosstab(x, y)

        ax = sns.heatmap(
            table, annot=annot, fmt="g"
        )
        ax.set_title(f"{x} vs {y}")
        plt.tight_layout()
        plt.show()
        return ax

    def clustermap(self, x, y, method="average"):
        """Cluster and plot a categorical frequency table."""
        table = self.crosstab(x, y)

        if table.empty or min(table.shape) < 2:
            raise ValueError(
                "Clustermap requires at least two rows "
                "and two columns in the frequency table."
            )

        result = sns.clustermap(
            table, annot=True, fmt="g", method=method
        )
        plt.show()
        return result

    def pairplot(self, columns=None, hue=None):
        """Plot pairwise relationships among numerical columns."""
        if columns is None:
            columns = self.df.select_dtypes(
                include="number"
            ).columns.tolist()

        if len(columns) < 2:
            raise ValueError(
                "Pairplot requires at least two numerical columns."
            )

        for column in columns:
            self._check_numeric(column)

        if hue is not None:
            self._check_column(hue)

        result = sns.pairplot(
            data=self.df,
            vars=columns,
            hue=hue
        )
        plt.show()
        return result

    def lineplot(self, x, y, hue=None):
        """Plot how a numerical value changes across x."""
        self._check_column(x)
        self._check_numeric(y)

        if hue is not None:
            self._check_column(hue)

        ax = sns.lineplot(
            data=self.df, x=x, y=y, hue=hue
        )
        ax.set_title(f"{y} by {x}")
        plt.tight_layout()
        plt.show()
        return ax

    def pivot_table(
        self, values, index, columns, aggfunc="mean"
    ):
        """Create a summary table with row and column groups."""
        self._check_numeric(values)
        self._check_column(index)
        self._check_column(columns)

        return pd.pivot_table(
            self.df,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc
        )

    def correlation(self, columns=None):
        """Calculate pairwise correlations of numerical columns."""
        if columns is None:
            numeric_df = self.df.select_dtypes(
                include="number"
            )
        else:
            for column in columns:
                self._check_numeric(column)

            numeric_df = self.df[columns]

        return numeric_df.corr()

    def group_summary(
        self, category, value, aggfunc="mean"
    ):
        """Aggregate a numerical column by category."""
        self._check_column(category)
        self._check_numeric(value)

        if aggfunc not in ("mean", "sum", "median", "min", "max", "count"):
            raise ValueError(
                "Unsupported aggfunc. Use mean, sum, median, "
                "min, max, or count."
            )

        return self.df.groupby(category)[value].agg(aggfunc)
