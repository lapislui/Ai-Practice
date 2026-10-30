from .help import Help


class DataDefinition(Help):
    methods = {
        # Object display methods
        "__str__": {
            "description": "Display a readable dataset summary.",
            "parameters": "None.",
            "returns": "String containing rows, columns, and column names.",
            "example": "print(data)"
        },
        "__repr__": {
            "description": "Display a short representation of the object.",
            "parameters": "None.",
            "returns": "String showing the number of rows and columns.",
            "example": "repr(data)"
        },

        # Special methods
        "__len__": {
            "description": "Get the number of rows in the dataset.",
            "parameters": "None.",
            "returns": "Integer representing the number of rows.",
            "example": "len(data)"
        },
        "__getitem__": {
            "description": "Select a column using its name.",
            "parameters": "column: name of the column.",
            "returns": "Pandas Series for the selected column.",
            "example": "data['Age']"
        },
        "__contains__": {
            "description": "Check whether a column exists.",
            "parameters": "column: name of the column.",
            "returns": "Boolean: True or False.",
            "example": "'Age' in data"
        },

        # Data definition methods
        "shape": {
            "description": "Get the number of rows and columns.",
            "parameters": "None.",
            "returns": "Tuple containing (rows, columns).",
            "example": "data.shape()"
        },
        "columns": {
            "description": "Get all column names.",
            "parameters": "None.",
            "returns": "List of column names.",
            "example": "data.columns()"
        },
        "sample": {
            "description": "Get random rows from the dataset.",
            "parameters": "n=5: number of rows to return.",
            "returns": "Pandas DataFrame containing sampled rows.",
            "example": "data.sample(3)"
        },
        "data_types": {
            "description": "Get the data type of each column.",
            "parameters": "None.",
            "returns": "Pandas Series containing column data types.",
            "example": "data.data_types()"
        },
        "info": {
            "description": "Display column names, data types, and non-null counts.",
            "parameters": "None.",
            "returns": "None; prints information directly.",
            "example": "data.info()"
        },
        "describe": {
            "description": "Get statistical summaries of columns.",
            "parameters": "None.",
            "returns": "DataFrame containing descriptive statistics.",
            "example": "data.describe()"
        },
        "missing_values": {
            "description": "Count missing values in each column.",
            "parameters": "None.",
            "returns": "Pandas Series containing missing-value counts.",
            "example": "data.missing_values()"
        },
        "missing_percentage": {
            "description": "Calculate the percentage of missing values in each column.",
            "parameters": "None.",
            "returns": "Pandas Series containing percentages.",
            "example": "data.missing_percentage()"
        },
        "duplicates": {
            "description": "Count duplicate rows in the dataset.",
            "parameters": "None.",
            "returns": "Integer representing the number of duplicate rows.",
            "example": "data.duplicates()"
        },
        "unique_values": {
            "description": "Count unique values in each column.",
            "parameters": "None.",
            "returns": "Pandas Series containing unique-value counts.",
            "example": "data.unique_values()"
        },
        "memory_usage": {
            "description": "Calculate the memory used by the dataset.",
            "parameters": "None.",
            "returns": "Memory usage in KB, rounded to two decimal places.",
            "example": "data.memory_usage()"
        },

        # Inherited help method
        "help": {
            "description": "Display help for all methods or one specific method.",
            "parameters": "method=None: optional method name.",
            "returns": "None; prints method documentation.",
            "example": "data.help()\ndata.help('sample')"
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
        # print("Shape is")
        return f"Shae is {self.df.shape}"

    def columns(self):
        return self.df.columns.tolist()

    def sample(self, n=5):
        print("Sample is ")
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

