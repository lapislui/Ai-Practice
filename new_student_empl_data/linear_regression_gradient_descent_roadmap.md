# Linear Regression Practice Roadmap

## 1. Which dataset should you use?

For learning **Simple Linear Regression using Gradient Descent**, start with the student dataset, not the combined student-employee dataset.

The best way to learn is to use one input feature \(X\), one target \(y\), and then measure how well your model predicts the target. After that, expand to multiple linear regression and compare the metrics.

| File | Best use |
|---|---|
| `student_ml_linked.csv` | **Start here:** simple and multiple linear regression on academic data |
| `employee_data_linked.csv` | Later: salary prediction using employee features |
| `student_employee_analysis.csv` | Later: combined analysis, relationships between education and employment |
| `student_employee_ml_practice.xlsx` | Explore all datasets and the data dictionary in one workbook |

## 2. Start with Simple Linear Regression

Use the student CSV and select:

- **Input \(X\):** `study_hours_per_day`
- **Target \(y\):** `final_exam_score`

Your model learns a line that predicts exam scores from study hours.

The equation is:

\[
\hat y=b_0+b_1x
\]

- \(b_0\): intercept — the predicted score when study hours are zero.
- \(b_1\): slope — the predicted change in score for each additional study hour.
- \(\hat y\): predicted exam score.

Gradient descent repeatedly updates \(b_0\) and \(b_1\) to reduce prediction error.

## 3. Regression Metrics to Calculate

After training your model, measure its performance on a **test set the model has never seen**.

| Metric | What it tells you | Better result |
|---|---|---|
| MAE (Mean Absolute Error) | Average absolute prediction error | Lower |
| MSE (Mean Squared Error) | Average squared prediction error; penalizes large errors | Lower |
| RMSE (Root Mean Squared Error) | Error in the same units as exam scores | Lower |
| \(R^2\) (R-squared) | How much variation in scores the model explains relative to predicting the mean | Usually closer to 1 |
| Adjusted \(R^2\) | Adjusts \(R^2\) for the number of input features | Useful for comparing multiple regression models |

For example, an RMSE of 5 means the model's error scale is about 5 exam-score points. It does **not** mean every prediction is within 5 points.

## 4. Learning Roadmap

### Step 1 — Simple Linear Regression

**Input:** `study_hours_per_day`  
**Target:** `final_exam_score`

Implement gradient descent manually. Calculate residuals, MSE, MAE, RMSE, and \(R^2\).

### Step 2 — Multiple Linear Regression

Predict `final_exam_score` using features such as:

- Study hours
- Attendance
- Previous exam score
- Practice tests

Use gradient descent with multiple coefficients and compare the metrics against your simple model.

### Step 3 — Employee Salary Regression

**Input:** `ExperienceYears`  
**Target:** `AnnualSalaryINR`

Use the employee CSV after checking salary units and unusual values. Validate the data before modeling if salaries look unusually large.

### Step 4 — Model Comparison

Compare your manually implemented gradient descent model with scikit-learn's `LinearRegression`.

Use the same train/test split and metrics for a fair comparison.

## 5. Prepare Your First Experiment

Use `student_ml_linked.csv` and these settings:

- **Input feature:** `study_hours_per_day`
- **Target:** `final_exam_score`
- **Test size:** 20%
- **Random seed:** 42
- **Model:** Simple linear regression trained with gradient descent
- **Loss function:** Mean squared error
- **Learning rate:** Start with `0.01`, adjusting if needed
- **Feature scaling:** Optional for this single feature, but useful to learn for later models

### Recommended workflow

1. Load the CSV using pandas.
2. Inspect column names, data types, missing values, and summary statistics.
3. Plot `study_hours_per_day` against `final_exam_score`.
4. Split the data into training and test sets.
5. Fit the model using gradient descent on the training data only.
6. Plot the fitted line against the training data.
7. Calculate residuals and plot them.
8. Evaluate the final model on the held-out test set using MAE, MSE, RMSE, and \(R^2\).
9. Compare the results with scikit-learn's `LinearRegression` using the same split.

## 6. Regression vs. Classification

Use **regression metrics** when predicting continuous values, such as exam scores or salaries.

When predicting `Hired` or `passed_exam`, use **classification metrics**, such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 7. Important Modeling Notes

- Do not use `StudentID` or `EmployeeID` as predictive features. They are identifiers, not meaningful measurements.
- When predicting hiring, exclude features that are only available after hiring; otherwise, data leakage can make the model appear better than it really is.
- Fit any scaling or preprocessing steps on the training data only, then apply them to the test data.
- Keep the test set untouched until final evaluation.
- A low training error does not guarantee good predictions on unseen data.
- Adjusted \(R^2\) is mainly useful when comparing multiple linear regression models fitted to the same target and dataset. It is not needed for simple linear regression with one feature.

## Recommended Starting Point

Master the first experiment completely: understand the gradient updates, convergence, residual plot, and core regression metrics. Then move to multiple linear regression.
