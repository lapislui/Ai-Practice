# Add Simple Linear Regression to `AI_data`

This bundle contains two Python files:

- `simple_linear_regression.py` — a reusable `SimpleLinearRegression` class trained with batch gradient descent.
- `run_simple_linear_regression.py` — an example that uses the existing `AI_data` package and the student dataset.

## 1. Add the module to your existing package

Copy `simple_linear_regression.py` into your existing `AI_data/` directory. Do not replace the other package files.

Add this import to `AI_data/__init__.py`:

```python
from .simple_linear_regression import SimpleLinearRegression
```

Keep all existing imports in `__init__.py`; add this line alongside them. This makes both imports possible:

```python
import AI_data as aid
model = aid.SimpleLinearRegression()
```

The example script imports the new class directly from its module so it also works if you have not yet added the export line.

## 2. Place and run the example

Your project should look approximately like this:

```text
your-project/
├── AI_data/
│   ├── __init__.py
│   ├── help.py
│   ├── load_data.py
│   ├── data_definition.py
│   ├── univariate.py
│   ├── bivariate.py
│   ├── Standarddeviation.py
│   └── simple_linear_regression.py
├── student_ml_linked.csv
└── run_simple_linear_regression.py
```

Run from the project root:

```bash
python run_simple_linear_regression.py
```

The script expects the student CSV to contain `study_hours_per_day` and `final_exam_score`. If your actual file has different column names, change `FEATURE` and `TARGET` near the top of the script.

## 3. What the model provides

- `fit(X, y, learning_rate=0.01, epochs=2000, tolerance=1e-10)`
- `predict(X)`
- `evaluate(X, y_true)`
- `metrics(y_true, y_pred)`
- `residuals(X, y_true)`
- `equation()`
- `plot_regression(X, y)`
- `plot_loss()`
- `plot_residuals(X, y_true)`
- `help()` / `help("fit")`, inherited from the existing help base class when available

The model intentionally accepts **one feature only**. This keeps it a true simple linear regression implementation and prevents accidentally passing multiple columns.

## 4. Metrics

The evaluation returns:

- **MAE** — mean absolute error; lower is better.
- **MSE** — mean squared error; lower is better.
- **RMSE** — square root of MSE, in the target's units; lower is better.
- **R²** — coefficient of determination; 1 is a perfect fit, 0 is equivalent to predicting the test-target mean, and a negative score means worse than that baseline.
- **Adjusted R²** — adjusts R² for the number of features; for simple regression there is one feature. It is undefined when there are too few observations.

A test-set score can be negative. That is valid and indicates poor generalization, not a calculation failure.

## 5. Gradient descent notes

For each epoch, the model updates the weight and bias using the gradients of mean squared error. It internally standardizes the feature and target for numerical stability, then converts the final slope and intercept back to the original units. The loss history is therefore the **scaled-space training MSE**, not the original target-unit MSE.

The test set is kept separate from training. The model does not use scikit-learn to fit regression; scikit-learn is only used indirectly by the existing package's optional train/test splitting helper.

## Dependencies

The existing `AI_data` package already uses pandas, NumPy, Matplotlib, and scikit-learn in its documented setup. If needed:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```
