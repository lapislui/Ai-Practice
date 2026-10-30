# Simple Linear Regression for AI_data

This version follows the lightweight usage style of your existing package:

```python
import AI_data as aid

loader = aid.LoadData("employee_data_practice.csv")
df = loader.read()

reg = aid.SimpleLinearRegression(df)
reg.fit("experience_years", "salary")
reg.summary()

print(reg.predict(5))
reg.plot()
```

## Add it without replacing your package

1. Copy `simple_linear_regression.py` into the existing `AI_data/` package folder.
2. Open the existing `AI_data/__init__.py`.
3. Add this import on a new line. Keep all existing imports:

```python
from .simple_linear_regression import SimpleLinearRegression
```

4. Install dependencies if needed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

5. Put `run_simple_linear_regression.py` beside your CSV and run:

```bash
python run_simple_linear_regression.py
```

## Available methods

- `reg.fit("x_column", "y_column")` trains the model.
- `reg.predict(5)` predicts one value; `reg.predict([1, 2, 3])` predicts several.
- `reg.summary()` prints the equation and test metrics.
- `reg.evaluate()` returns MAE, MSE, RMSE, R² and adjusted R².
- `reg.plot()` displays the data and regression line.
- `reg.loss_plot()` displays training loss.
- `reg.residual_plot()` displays test residuals.
- `reg.help()` or `reg.help("predict")` shows usage help.

## Algorithm

The model uses scikit-learn's `SGDRegressor`, which trains using stochastic gradient descent. `StandardScaler` scales the input and target during training; predictions and the displayed equation are converted back to the original units. A test split is held out for evaluation.

Change `"experience_years"` and `"salary"` to the actual column names in your CSV. You can check them with `print(df.columns)`.
