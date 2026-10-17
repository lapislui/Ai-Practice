from .help import Help


class DataDefinition(Help):
    methods = {
        "shape": {
            "description": "Get the number of rows and columns.",
            "parameters": "None.",
            "returns": "Tuple of rows and columns.",
            "example": "data.shape()"
        },
        "columns": {
            "description": "Get all column names.",
            "parameters": "None.",
            "returns": "List of column names.",
            "example": "data.columns()"
        },
        "sample": {
            "description": "Get random rows.",
            "parameters": "n=5: number of rows.",
            "returns": "Pandas DataFrame.",
            "example": "data.sample(3)"
        },
        "missing_values": {
            "description": "Count missing values per column.",
            "parameters": "None.",
            "returns": "Pandas Series.",
            "example": "data.missing_values()"
        }
    }

    def __init__(self, df):
        self.df = df

    # Display dataset summary
    def __str__(self):
        return (
            f"DataDefinition\n"
            f"Rows: {self.df.shape[0]}\n"
            f"Columns: {self.df.shape[1]}\n"
            f"Column names: {self.df.columns.tolist()}"
        )

    def __repr__(self):
        return (
            f"DataDefinition(rows={len(self.df)}, "
            f"columns={len(self.df.columns)})"
        )

    # Special methods
    def __len__(self):
        return len(self.df)

    def __getitem__(self, column):
        return self.df[column]

    def __contains__(self, column):
        return column in self.df.columns

    # Data definition methods
    def shape(self):
        return self.df.shape

    def columns(self):
        return self.df.columns.tolist()

    def sample(self, n=5):
        return self.df.sample(
            n=min(n, len(self.df)),
            random_state=42
        )

    def data_types(self):
        return self.df.dtypes

    def info(self):
        return self.df.info()

    def describe(self):
        return self.df.describe(include="all").T

    def missing_values(self):
        return self.df.isnull().sum()

    def missing_percentage(self):
        return (self.df.isnull().mean() * 100).round(2)

    def duplicates(self):
        return self.df.duplicated().sum()

    def unique_values(self):
        return self.df.nunique()

    def memory_usage(self):
        return round(
            self.df.memory_usage(deep=True).sum() / 1024, 2
        )

