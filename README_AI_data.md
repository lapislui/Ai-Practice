# AI_data

**AI_data** is a beginner-friendly Python module for exploring datasets with pandas, NumPy, Matplotlib, and Seaborn. It organizes data loading, dataset inspection, univariate analysis, bivariate analysis, and standard-deviation/standardization tools into reusable classes.

## Features

- Load CSV files from a local path or HTTP(S) URL.
- Inspect rows, columns, data types, missing values, duplicates, unique values, and memory usage.
- Perform univariate analysis with statistics and plots.
- Explore relationships between variables with bivariate analysis.
- Calculate mean, variance, standard deviation, and Z-scores without scikit-learn.
- Standardize selected or all numerical columns.
- Use a shared `help()` method to discover documented methods.

## Project structure

```text
matrix practice/
├── AI_data/
│   ├── __init__.py
│   ├── help.py
│   ├── load_data.py
│   ├── data_definition.py
│   ├── univariate.py
│   ├── bivariate.py
│   └── Standarddeviation.py
├── employee_data_practice.csv
└── AI_testing.py
```

**Important:** Python module filenames are case-sensitive on some systems. Match the import in `AI_data/__init__.py` to the exact filename. If your file is called `standard_deviation.py`, import from `.standard_deviation` instead of `.Standarddeviation`.

## Requirements

Install the required libraries in your active environment:

```bash
python -m pip install pandas numpy matplotlib seaborn
```

The standard-deviation module described here does not require `sklearn.preprocessing.StandardScaler`.

## Setup

In `AI_data/__init__.py`, export the classes using imports that match your actual filenames. For the structure above:

```python
from .help import Help
from .load_data import LoadData
from .data_definition import DataDefinition
from .univariate import Univariate
from .bivariate import Bivariate
from .Standarddeviation import StandardDeviation
```

Use the package in your script or notebook:

```python
import AI_data as aid
```

## Quick start

Place your CSV next to your test script, outside the `AI_data` folder:

```python
import AI_data as aid

loader = aid.LoadData("employee_data_practice.csv")
df = loader.read()

data = aid.DataDefinition(df)
print(data)
print(data.shape())
print(data.columns())
print(data.data_types())
print(data.missing_values())
print(data.describe())
```

`LoadData` resolves a relative CSV path from the parent of the `AI_data` package directory. It can also load a CSV from an HTTP(S) URL.

## Shared help system

Analysis classes inherit from `Help` and document their public methods in a class-level `methods` dictionary.

```python
sd = aid.StandardDeviation(df)

sd.help()                    # List documented methods
sd.help("summary")           # Details about one method
```

## Modules

### `help.py` — `Help`

Provides the shared `help(method=None)` method. Each inheriting class supplies method descriptions, parameters, return values, and examples in its `methods` dictionary.

### `load_data.py` — `LoadData`

Loads a CSV file into a pandas DataFrame.

```python
loader = aid.LoadData("employee_data_practice.csv")
df = loader.read()
```

### `data_definition.py` — `DataDefinition`

Inspects the dataset and provides:

- `shape()` — number of rows and columns
- `columns()` — list of column names
- `sample(n=5)` — sample rows
- `data_types()` — column data types
- `info()` — non-null counts and types
- `describe()` — descriptive statistics
- `missing_values()` and `missing_percentage()`
- `duplicates()` — duplicate row count
- `unique_values()` — unique count per column
- `memory_usage()` — approximate memory usage in KB

It also implements Python special methods including `__str__`, `__repr__`, `__len__`, `__getitem__`, and `__contains__`.

### `univariate.py` — `Univariate`

Analyzes one variable at a time. The current implementation includes value counts, summary, minimum, maximum, mean, median, mode, standard deviation, variance, skewness, kurtosis, countplots, pie charts, histograms, and boxplots.

```python
uni = aid.Univariate(df)
uni.help()
print(uni.mean("Age"))
print(uni.summary("Age"))
uni.histogram("Age")
uni.boxplot("Age")
```

Use column names that exist in your dataset.

### `bivariate.py` — `Bivariate`

Explores relationships between variables. The current implementation includes scatterplots, barplots, boxplots, KDE plots, crosstabs, heatmaps, clustermaps, pairplots, lineplots, pivot tables, correlation, and grouped summaries.

```python
bi = aid.Bivariate(df)
bi.help()
print(bi.crosstab("Department", "JobLevel"))
print(bi.correlation())
print(bi.group_summary("Department", "AnnualSalaryINR"))
bi.scatterplot("Age", "AnnualSalaryINR")
```

Check each method's help for column requirements and valid combinations of categorical and numerical data.

### `Standarddeviation.py` — `StandardDeviation`

Calculates statistics and standardizes numeric data using pandas and NumPy—without scikit-learn.

```python
sd = aid.StandardDeviation(df)

# All numerical columns
print(sd.mean())
print(sd.variance())
print(sd.standard_deviation())
print(sd.summary())
print(sd.z_score())

# One column
print(sd.mean("Age"))
print(sd.standard_deviation("AnnualSalaryINR"))
print(sd.summary("Age"))

# Standardize all numerical columns
scaled_df = sd.standardize()
print(scaled_df.head())

# Standardize selected columns
scaled_selected = sd.standardize(["Age", "AnnualSalaryINR"])

# Plot one or all numerical columns
sd.histogram()
sd.histogram("AnnualSalaryINR")
sd.boxplot()
sd.boxplot("Age")
```

#### Sample and population statistics

The `variance()`, `standard_deviation()`, `z_score()`, and `standardize()` methods accept `population=False` by default.

- `population=False`: sample variance/standard deviation, using `ddof=1`.
- `population=True`: population variance/standard deviation, using `ddof=0`.

Use the same convention when comparing results.

#### Missing values and outliers

Pandas statistical calculations skip missing values. Z-scores for a constant or otherwise zero-variance column are returned as missing values. `standardize()` raises an error if a selected column has zero or undefined standard deviation.

Standardization changes the scale of numerical values. It does **not** make a distribution normal and does **not** automatically remove outliers.

## Employee dataset example

`employee_data_practice.csv` is synthetic learning data, not real employee information. It contains employee-like fields such as age, department, experience, annual salary, performance rating, satisfaction, weekly hours, and leave days. Some extreme values and missing values are intentionally included for practice.

```python
import AI_data as aid
import matplotlib.pyplot as plt

df = aid.LoadData("employee_data_practice.csv").read()
sd = aid.StandardDeviation(df)

print(sd.summary())

scaled_df = sd.standardize()

# Histograms of all numeric columns after standardization
scaled_df.select_dtypes(include="number").hist(
    figsize=(15, 12),
    bins=20,
    edgecolor="black"
)
plt.suptitle("Standardized Employee Data")
plt.tight_layout()
plt.show()
```

## Suggested learning workflow

1. Load your CSV with `LoadData`.
2. Inspect the dataset with `DataDefinition`.
3. Check data types, missing values, duplicates, and unusual values.
4. Use `Univariate` to analyze individual columns.
5. Use `Bivariate` to explore relationships between columns.
6. Use `StandardDeviation` to measure spread, calculate Z-scores, and standardize numerical data.
7. Compare the original dataset with the transformed dataset before drawing conclusions.

## Troubleshooting

### `summary()` says a column argument is missing

Check that your method definition uses `def summary(self, column=None, population=False):`. Restart the Jupyter kernel after editing the module, then rerun your import/setup cells. Python may still be using the old class in memory.

### `AI_data` has no attribute `StandardDeviation`

Check that `AI_data/__init__.py` imports the class from the exact module filename, then restart the kernel.

### A numeric-looking column is treated as text

Inspect `df.dtypes`. Convert values to numeric only after checking the source data, for example with `pd.to_numeric(df["Column"], errors="coerce")`.

## Project status

This README describes the learning-oriented module API. Update it whenever you add, remove, or rename public methods.
