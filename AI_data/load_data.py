import pandas as pd
from pathlib import Path
from .help import Help


class LoadData(Help):
    methods = {
        "read": {
            "description": "Load a CSV file into a DataFrame.",
            "parameters": "Source is provided during initialization.",
            "returns": "Pandas DataFrame.",
            "example": "loader.read()"
        }
    }

    def __init__(self, source):
        self.source = source

    def read(self):
        if self.source.startswith(("http://", "https://")):
            return pd.read_csv(self.source)

        path = Path(__file__).resolve().parent.parent / self.source
        return pd.read_csv(path)