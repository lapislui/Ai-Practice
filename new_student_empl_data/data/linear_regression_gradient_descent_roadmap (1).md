# Machine Learning Roadmap: Learn Statistics and Regression by Coding

A practical, beginner-friendly roadmap for the four supplied practice files. It is written for someone who can write Python but does **not** yet feel comfortable with statistics.

The goal is not to memorize formulas. The goal is to understand what each number means, see it in a plot, implement it in Python, and then use scikit-learn to check your implementation.

---

## 0. Your four files: what each one is for

You have three CSV files and one Excel workbook. The workbook is a convenient container for the same core datasets plus a dictionary explaining the columns.

| File | Size | What is inside | Best use |
|---|---:|---|---|
| `student_ml_linked.csv` | 600 rows × 14 columns | Student study habits, exam scores, pass status, and hiring link | **Start here.** Regression and classification |
| `employee_data_linked.csv` | 250 rows × 21 columns | Employee profile, experience, salary, performance, satisfaction, and education history | Employee analysis and regression practice |
| `student_employee_analysis.csv` | 600 rows × 27 columns | Student records joined to employee details where a match exists | Join analysis, missing values, and careful feature selection |
| `student_employee_ml_practice.xlsx` | 5 sheets | `Student_Data`, `Employee_Data`, `Joined_Analysis`, `Data_Dictionary`, `Start_Here` | Browse all data and understand column meanings |

### Important facts found in the supplied files

- The student CSV has 600 students. The `Hired` column contains 250 ones and 350 zeros.
- The employee CSV has 250 employee records.
- The joined analysis has 600 rows. Employee columns are blank for the 350 students without a matching employee record. Those blanks are expected for students who were not hired; they are not automatically bad data.
- Some student measurement columns have a small number of missing values. Inspect each column before training.
- `StudentID` and `EmployeeID` are identifiers / join keys. Do not feed them into a prediction model as if they were meaningful measurements.
- **Important salary warning:** in the supplied employee CSV, `AnnualSalaryINR` is `180000000` for every one of the 250 rows. It has no variation. A model cannot learn to predict differences in salary from a target that never changes. Do not use this column as a meaningful regression exercise until the data is corrected or replaced.
- The workbook's `Data_Dictionary` sheet describes the columns and includes practice notes. Check it when unsure what a field means.

The files are synthetic practice data. Use them to learn the workflow and mathematics, not to make real-world claims about students, hiring, or salaries.

---

# Part 1 — Learn to look at data before modeling

## Step 1. Load and inspect the student data

Use the style of your `AI_data` package:

```python
import AI_data as aid

loader = aid.LoadData("student_ml_linked.csv")
df = loader.read()

df.info()
print(df.head())
print(df.columns)
print(df.isna().sum())
print(df.describe())
```

What these commands mean:

- `head()` shows a few rows so you can see what one record looks like.
- `columns` lists the field names.
- `info()` shows how many values each column has and what type Python thinks they are.
- `isna().sum()` counts blanks in each column.
- `describe()` gives a quick numeric summary: count, average, spread, smallest and largest values.

### The statistics hiding inside `describe()`

Think of a column as a list of measurements.

- **Count:** how many usable measurements exist.
- **Mean:** the balancing-point average. Add the values and divide by the number of values.
- **Median:** the middle value after sorting. It is less affected by extreme values.
- **Standard deviation:** roughly how spread out the values are around their average.
- **Min / max:** the smallest and largest values.

Try:

```python
print(df["study_hours_per_day"].mean())
print(df["study_hours_per_day"].median())
print(df["study_hours_per_day"].std())
```

Do not just print these values. Ask: are the mean and median close? Is the standard deviation small or large compared with the average? That is how you start understanding a dataset.

## Step 2. Plot before you predict

```python
import matplotlib.pyplot as plt

plt.scatter(df["study_hours_per_day"], df["final_exam_score"])
plt.xlabel("Study hours per day")
plt.ylabel("Final exam score")
plt.title("Study time vs final exam score")
plt.show()
```

A scatter plot puts one student at one dot.

- A cloud rising from left to right suggests a positive relationship.
- A cloud falling from left to right suggests a negative relationship.
- A shapeless cloud suggests a weak or unclear straight-line relationship.
- A few dots far away from the rest may be unusual observations worth investigating.

A relationship is not proof of cause. A plot can show association; it cannot prove that studying alone caused a score.

## Step 3. Understand correlation

```python
r = df["study_hours_per_day"].corr(df["final_exam_score"])
print("Correlation:", r)
```

Correlation is a number from -1 to +1:

- Near `+1`: the two measurements tend to rise together.
- Near `-1`: when one rises, the other tends to fall.
- Near `0`: there is little straight-line association.

Correlation is not the slope of your regression line, and correlation does not prove causation.

---

# Part 2 — Your first model: Simple Linear Regression

## Step 4. Understand the job of regression

Use:

- Input `X`: `study_hours_per_day`
- Target `y`: `final_exam_score`

Imagine you want to estimate an exam score from study hours. The model draws a straight line through the scatter plot.

The equation is:

\[
\hat{y} = b_0 + b_1x
\]

Read it like ordinary code:

```python
predicted_score = intercept + slope * study_hours
```

- `x` is the study hours.
- `b0` is the **intercept**: where the line predicts a score when study hours are zero.
- `b1` is the **slope**: how much the prediction changes when study time increases by one hour.
- `y_hat` is the model's predicted score.

The intercept is a mathematical part of the line. It does not automatically mean that a student who studies zero hours would really receive that score.

## Step 5. A prediction is not the actual value

Suppose a student's actual score is 72 and your model predicts 68:

```python
actual = 72
predicted = 68

error = actual - predicted
print(error)  # 4
```

This difference is called the **residual**:

\[
e = y - \hat{y}
\]

- Positive residual: prediction was too low.
- Negative residual: prediction was too high.
- Zero: exact prediction.

The model tries to make these errors small overall.

## Step 6. Understand the loss function (MSE)

For training, a common choice is Mean Squared Error:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Plain-English version:

1. Predict every training student's score.
2. Subtract prediction from actual score.
3. Square each error so negative and positive errors do not cancel, and large errors count more.
4. Average the squared errors.

Example:

```python
actual = [70, 80, 90]
predicted = [68, 83, 88]

errors = [a - p for a, p in zip(actual, predicted)]
squared_errors = [e ** 2 for e in errors]
mse = sum(squared_errors) / len(squared_errors)

print(errors)          # [2, -3, 2]
print(squared_errors)  # [4, 9, 4]
print(mse)             # 5.666...
```

MSE is measured in squared score units, which is not very intuitive. That is why RMSE is also useful later.

---

# Part 3 — Gradient Descent, without the scary statistics language

## Step 7. Think of the line as two adjustable knobs

Your line has two values:

- `b0`: move the line up or down.
- `b1`: rotate the line to make it steeper or flatter.

At first, the values can be guesses. Gradient descent repeatedly nudges these knobs to reduce MSE.

Imagine standing on a hill in fog and trying to get to the lowest point. You feel which direction slopes downward, take a step, and repeat. That is the basic idea of gradient descent.

## Step 8. The gradient tells us which way to adjust

For simple linear regression:

\[
\hat{y}_i = b_0 + b_1x_i
\]

\[
MSE = \frac{1}{n}\sum(\hat{y}_i-y_i)^2
\]

The derivatives (the gradients) are:

\[
\frac{\partial MSE}{\partial b_0}
= \frac{2}{n}\sum(\hat{y}_i-y_i)
\]

\[
\frac{\partial MSE}{\partial b_1}
= \frac{2}{n}\sum x_i(\hat{y}_i-y_i)
\]

You do not need to fear the derivative notation. Read it as:

- The first formula measures how the loss changes if you adjust the intercept.
- The second measures how the loss changes if you adjust the slope.
- The `x` in the slope formula matters because changing the slope affects points differently depending on their x value.

The update rule is:

\[
b_0 \leftarrow b_0 - \alpha \frac{\partial MSE}{\partial b_0}
\]

\[
b_1 \leftarrow b_1 - \alpha \frac{\partial MSE}{\partial b_1}
\]

`alpha` is the **learning rate**: how large a step to take.

- Too large: you may jump over the bottom and the loss may bounce or grow.
- Too small: learning can be very slow.
- A reasonable value: loss generally decreases and eventually levels off.

## Step 9. Implement gradient descent yourself

This is the learning version. It makes every operation visible.

```python
import numpy as np

# Select input and target.
X = df["study_hours_per_day"].to_numpy(dtype=float)
y = df["final_exam_score"].to_numpy(dtype=float)

# Remove rows where either selected value is missing.
valid = np.isfinite(X) & np.isfinite(y)
X = X[valid]
y = y[valid]

# Start with a flat line at zero.
b0 = 0.0
b1 = 0.0

learning_rate = 0.01
epochs = 2000
n = len(X)

loss_history = []

for epoch in range(epochs):
    # 1. Predict using the current line.
    y_pred = b0 + b1 * X

    # 2. Calculate prediction errors.
    errors = y_pred - y

    # 3. Calculate the two gradients.
    db0 = (2 / n) * np.sum(errors)
    db1 = (2 / n) * np.sum(X * errors)

    # 4. Move both parameters in the downhill direction.
    b0 = b0 - learning_rate * db0
    b1 = b1 - learning_rate * db1

    # 5. Record loss so we can see whether learning is working.
    new_predictions = b0 + b1 * X
    mse = np.mean((y - new_predictions) ** 2)
    loss_history.append(mse)

print("Intercept:", b0)
print("Slope:", b1)
print("Final MSE:", loss_history[-1])
```

### What each loop iteration is doing

| Code | Human explanation |
|---|---|
| `y_pred = b0 + b1 * X` | Draw the current line and predict scores |
| `errors = y_pred - y` | See how far each prediction is from reality |
| `db0 = ...` | Calculate how the intercept should move |
| `db1 = ...` | Calculate how the slope should move |
| `b0 = b0 - ...` | Adjust the line up/down |
| `b1 = b1 - ...` | Adjust the line's steepness |
| `loss_history.append(mse)` | Save the current error so we can inspect learning |

### Plot the loss

```python
plt.plot(loss_history)
plt.xlabel("Training iteration")
plt.ylabel("MSE")
plt.title("Did gradient descent learn?")
plt.show()
```

A healthy run usually trends downward and then becomes flatter. If it explodes or becomes `nan`, the learning rate may be too high or the feature scale may be difficult for the chosen learning rate.

**Scaling note:** study hours and scores are on different numeric ranges. For later experiments, learn `StandardScaler`. Fit the scaler on training data only, then use that fitted scaler to transform test data. Scaling changes the numeric representation, not the meaning of the data.

---

# Part 4 — Evaluate the model like an engineer

## Step 10. Split training and test data

Do not train and evaluate on exactly the same rows. If you do, you are asking the model questions it has already practiced.

- **Training set:** examples the model is allowed to learn from.
- **Test set:** examples kept aside to check how it performs on unseen records.

```python
from sklearn.model_selection import train_test_split

X = df[["study_hours_per_day"]]
y = df["final_exam_score"]

# For a first experiment, remove rows missing either selected field.
clean = df[["study_hours_per_day", "final_exam_score"]].dropna()
X = clean[["study_hours_per_day"]]
y = clean["final_exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

- `test_size=0.20` means 20% is held out for testing.
- `random_state=42` makes the split repeatable, so you can compare experiments fairly.
- The test set must not be used to update model parameters.

For the manual gradient descent experiment, train using `X_train` and `y_train`, then calculate metrics using `X_test` and `y_test`. Do not use all rows in the training loop after making the split.

## Step 11. Understand the main metrics

Use the same held-out test data for every model you compare.

### MAE — Mean Absolute Error

\[
MAE = \frac{1}{n}\sum |y_i-\hat{y}_i|
\]

It answers: **on average, how many score points away are the predictions?**

If MAE is 4, the average absolute miss is 4 exam-score points.

### MSE — Mean Squared Error

\[
MSE = \frac{1}{n}\sum(y_i-\hat{y}_i)^2
\]

It punishes large mistakes more heavily because errors are squared.

### RMSE — Root Mean Squared Error

\[
RMSE = \sqrt{MSE}
\]

It puts the error back into the target's units. An RMSE of 5 means the model's error scale is about 5 score points; it does **not** mean every prediction is within 5 points.

### R² — R-squared

R² compares your model against a very basic baseline: predicting the test-set average for every student.

- `1.0`: perfect predictions.
- `0.0`: no better than predicting the average (in the standard definition).
- Below `0`: worse than that average baseline.

R² is not “percentage of predictions that are correct.”

### Adjusted R²

Adjusted R² penalizes adding more input columns. It is mainly useful when comparing multiple linear regression models with the same target and dataset. It is not needed for your first one-feature model.

### Calculate metrics in code

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

# y_test = actual scores from the test set
# y_pred = predictions made for X_test
y_pred = b0 + b1 * X_test["study_hours_per_day"].to_numpy()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
```

If your manual model was trained on scaled X, use the same training-fitted scaler for test data and convert predictions back to the original score units before calculating these metrics.

## Step 12. Plot residuals

```python
residuals = y_test.to_numpy() - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted exam score")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual plot")
plt.show()
```

A residual plot is a model debugging tool.

- Random-looking scatter around zero is generally encouraging.
- A curve may mean a straight line is too simple.
- A funnel shape may mean prediction errors grow for some score ranges.
- A few very distant points deserve inspection.

---

# Part 5 — Use scikit-learn as a reference implementation

Once you understand the manual algorithm, compare it with scikit-learn. This is not cheating; it is how you check your implementation against a well-tested library.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

reference_model = LinearRegression()
reference_model.fit(X_train, y_train)

sklearn_predictions = reference_model.predict(X_test)

print("Intercept:", reference_model.intercept_)
print("Slope:", reference_model.coef_[0])
print("MAE:", mean_absolute_error(y_test, sklearn_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, sklearn_predictions)))
print("R2:", r2_score(y_test, sklearn_predictions))
```

`LinearRegression` uses a direct least-squares solution; it is a useful reference for the line that minimizes squared error. It is **not** the same training algorithm as your gradient descent loop.

For a gradient-descent comparison, use `SGDRegressor`:

```python
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sgd_model = make_pipeline(
    StandardScaler(),
    SGDRegressor(
        loss="squared_error",
        max_iter=5000,
        tol=1e-5,
        random_state=42
    )
)

sgd_model.fit(X_train, y_train)
sgd_predictions = sgd_model.predict(X_test)

print("SGD MAE:", mean_absolute_error(y_test, sgd_predictions))
print("SGD RMSE:", np.sqrt(mean_squared_error(y_test, sgd_predictions)))
print("SGD R2:", r2_score(y_test, sgd_predictions))
```

Do not expect every algorithm to produce exactly identical values. Compare the same split, target, features, and metrics.

---

# Part 6 — Move through all four supplied files

## Dataset A: `student_ml_linked.csv`

<badge color="success">Start here</badge>

### Experiment A1 — One feature predicts exam score

- X: `study_hours_per_day`
- y: `final_exam_score`
- Model: simple linear regression
- Learn: scatter plot, slope, intercept, residual, MSE, gradient descent, MAE, RMSE, R².

### Experiment A2 — Try other single features

Keep the target fixed and change only X:

- `attendance_percent` → `final_exam_score`
- `previous_exam_score` → `final_exam_score`
- `assignments_completed_percent` → `final_exam_score`
- `practice_tests_completed` → `final_exam_score`
- `sleep_hours_per_night` → `final_exam_score`

For each experiment, record the feature, correlation, test MAE, test RMSE, and test R². This teaches you that some inputs are more useful than others.

### Experiment A3 — Multiple Linear Regression

Now use several columns together:

```python
features = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_exam_score",
    "practice_tests_completed",
    "assignments_completed_percent"
]

X = df[features]
y = df["final_exam_score"]
```

The model becomes:

\[
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
\]

Each input gets its own coefficient. Each coefficient represents the model's estimated change in prediction when that feature increases by one unit **while the other included features are held fixed**.

Before training, handle missing values and split the data. Fit imputation and scaling on training data only. Compare its test metrics with Experiment A1.

Do not automatically assume more columns make a better model. Extra columns can add noise, overlap, or overfitting.

### Experiment A4 — Predict pass/fail (classification, not regression)

- Target: `passed_exam`
- Inputs: study habits and previous academic information.
- Model family: logistic regression or a decision tree classifier.
- Metrics: accuracy, precision, recall, F1-score, confusion matrix.

`passed_exam` is Yes/No, so it is a classification target. Do not evaluate it with MAE/RMSE as if it were a continuous score.

**Leakage warning:** `final_exam_score` is likely directly related to the pass/fail result. The workbook dictionary specifically warns against using it as a feature when predicting pass status if the pass result is defined from that score. Exclude it for a realistic prediction exercise.

### Experiment A5 — Predict hiring status

- Target: `Hired` (0/1) or `EmploymentStatus`.
- Start with student information available before hiring.
- Do not use `EmployeeID` or fields that only exist after the person is hired.
- Use classification metrics.

This asks: “Can information available about a student help estimate hiring status?” It does not prove why a person was hired.

## Dataset B: `employee_data_linked.csv`

### Experiment B1 — Explore employees

Try grouping and summarizing by:

- `Department`
- `JobLevel`
- `EmploymentType`
- `Education`
- `Location`

Examples:

```python
print(df["Department"].value_counts())
print(df.groupby("Department")["PerformanceRating"].mean())
print(df.groupby("JobLevel")["JobSatisfaction"].mean())
```

This is descriptive analysis: summarizing what is in the file, not predicting future outcomes.

### Experiment B2 — Predict performance rating

Possible target: `PerformanceRating`.

Possible numeric inputs to investigate:

- `ExperienceYears`
- `WeeklyHours`
- `LeaveDaysTaken`
- `Age`

You may later include categorical columns such as `Department` or `Education`, but they need encoding (for example, one-hot encoding). Do not pass text categories directly into a basic numeric regression model.

### Experiment B3 — Salary target: stop and validate first

The supplied `AnnualSalaryINR` column has the same value (`180000000`) in all 250 employee rows. That means:

- There is no salary variation to explain.
- A regression model cannot learn which employees earn more.
- A constant prediction can look deceptively perfect, but it is not a useful salary-prediction lesson.

Do not report a salary model as successful using this column. First correct the practice data or replace it with a dataset that has realistic varying salaries. Also verify the units and scale before modeling.

## Dataset C: `student_employee_analysis.csv`

This is the joined view: student information plus employee information where a student has a matching employee record.

### Experiment C1 — Understand the join

```python
df = aid.LoadData("student_employee_analysis.csv").read()

print(df.shape)
print(df.isna().sum().sort_values(ascending=False))
print(df[df["EmployeeID"].isna()].head())
```

For students who were not hired, employee fields such as department, experience, and salary are blank. That is expected from the relationship described in the workbook.

Do not fill every employee blank with zero. Zero years of experience and “not an employee record” mean different things.

### Experiment C2 — Compare student outcomes by employment status

Examples:

```python
print(df.groupby("EmploymentStatus")["final_exam_score"].mean())
print(df.groupby("EmploymentStatus")["attendance_percent"].mean())
```

These are group averages. They describe the supplied data; they do not establish that employment caused a difference.

### Experiment C3 — Avoid target leakage

If predicting `Hired`, do not use:

- `EmployeeID`
- `Department`
- `JobLevel`
- `EmploymentType`
- `ExperienceYears`
- `AnnualSalaryINR`
- `PerformanceRating`
- `JobSatisfaction`
- other fields that only exist after hiring

Those fields reveal the outcome or information created after it. A model could appear highly accurate simply because it has been given clues from the future.

Use pre-hiring student features only.

## Dataset D: `student_employee_ml_practice.xlsx`

The workbook has five sheets:

| Sheet | Purpose |
|---|---|
| `Student_Data` | Same student records as the student CSV |
| `Employee_Data` | Same employee records as the employee CSV |
| `Joined_Analysis` | Same joined analysis view |
| `Data_Dictionary` | Column meanings, data types, and practice notes |
| `Start_Here` | Explanation of how the datasets relate |

Use pandas to load the sheets:

```python
import pandas as pd

file = "student_employee_ml_practice.xlsx"

students = pd.read_excel(file, sheet_name="Student_Data")
employees = pd.read_excel(file, sheet_name="Employee_Data")
joined = pd.read_excel(file, sheet_name="Joined_Analysis")
dictionary = pd.read_excel(file, sheet_name="Data_Dictionary")
start = pd.read_excel(file, sheet_name="Start_Here")
```

Use the workbook as your reference copy and data dictionary. The CSV files are simpler for the first experiments.

---

# Part 7 — Statistics and ML vocabulary cheat sheet

| Word | Plain-English meaning |
|---|---|
| Observation / row | One student or one employee |
| Feature / input / X | Information given to the model |
| Target / label / y | What you want the model to predict |
| Mean | Arithmetic average |
| Median | Middle value after sorting |
| Variance | A measure of spread, based on squared distance from the mean |
| Standard deviation | Typical spread, expressed in the original unit |
| Correlation | Strength and direction of a straight-line relationship |
| Regression | Predicting a numeric value |
| Classification | Predicting a category |
| Coefficient / weight | How strongly an input changes the model's prediction |
| Intercept / bias | The model's baseline prediction when inputs are zero |
| Prediction | What the model estimates |
| Residual | Actual value minus predicted value |
| Loss / cost | A number measuring how wrong the model is during training |
| Gradient | Direction and rate in which loss changes as a parameter changes |
| Learning rate | Size of each parameter update |
| Epoch / iteration | A training pass or update cycle; exact meaning depends on the algorithm |
| Overfitting | Model learns training examples too closely and performs poorly on new data |
| Generalization | How well a model works on unseen examples |
| Data leakage | Training uses information it would not legitimately have at prediction time |
| Scaling | Changing numeric feature ranges to make optimization easier |
| Encoding | Converting categories such as `Department` into numeric inputs |

---

# Part 8 — Your practical learning schedule

Do not rush to neural networks. Finish these small projects in order.

| Stage | Build | You should be able to explain |
|---|---|---|
| 1 | Inspect student CSV | Rows, columns, types, missing values, mean, median, standard deviation |
| 2 | Scatter plot + correlation | Whether two variables move together |
| 3 | Manual simple regression | Slope, intercept, prediction and residual |
| 4 | Manual gradient descent | Gradients, learning rate, parameter updates and loss curve |
| 5 | Train/test evaluation | Why unseen test rows matter |
| 6 | Regression metrics | MAE, MSE, RMSE and R² |
| 7 | scikit-learn comparison | Manual gradient descent vs `LinearRegression` vs `SGDRegressor` |
| 8 | Multiple regression | Several features and several coefficients |
| 9 | Classification | Predict pass/fail or hiring status with classification metrics |
| 10 | Joined-data project | Joins, expected missing values and data leakage |
| 11 | Employee analysis | Group summaries and performance modeling; validate salary data first |

## A model experiment log

Keep a record instead of relying on memory:

| Experiment | Dataset | Features | Target | Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---:|---:|---:|
| 1 | Student | study hours | final score | Manual GD | | | |
| 2 | Student | attendance | final score | Manual GD | | | |
| 3 | Student | previous score | final score | Manual GD | | | |
| 4 | Student | 5 academic features | final score | Multiple regression | | | |

Use the same train/test split when comparing models on the same task. Do not compare scores from different targets as if they were directly equivalent.

---

# Final checklist: before saying “my model works”

- [ ] I know what each row represents.
- [ ] I know what my input columns mean.
- [ ] I chose a numeric target for regression.
- [ ] I checked missing values and unusual values.
- [ ] I excluded ID columns.
- [ ] I checked for target leakage.
- [ ] I split training and test data before fitting.
- [ ] I fitted preprocessing only on training data.
- [ ] I can explain what the slope and intercept mean.
- [ ] I can explain one prediction and its residual.
- [ ] I plotted the loss curve.
- [ ] I evaluated on the untouched test set.
- [ ] I can explain MAE, RMSE and R² in plain language.
- [ ] I compared against a scikit-learn reference model.
- [ ] I did not mistake correlation for causation.

**Your first mission:** use `student_ml_linked.csv`, predict `final_exam_score` from `study_hours_per_day`, and implement the gradient descent loop yourself. Do not move to multiple regression until you can explain every line in that loop in ordinary language.
