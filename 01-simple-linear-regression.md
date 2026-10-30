# Learn Simple Linear Regression: Theory, Statistics, and Every Notebook Step

Your notebook, `01-simple-linear-regression.ipynb`, teaches Simple Linear Regression using Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

We will study it as a statistics lesson, not just a coding exercise. For every step, you will learn:

- What the code does and why it is necessary.
- The mathematical formula behind it.
- The statistical meaning of each variable and calculation.
- What the graph represents and how to interpret it.
- What the model learns, how predictions work, and where they can go wrong.

One important distinction: your notebook uses Scikit-learn's `LinearRegression`, which fits a line using least squares. It does not manually implement gradient descent. We will understand the statistical objective first, then connect it to gradient descent so you understand both methods.

## 1. The entire process at a glance

Step 1 — Load the data

Read CGPA and package observations

Step 2 — Explore the relationship

Plot the observed data and inspect its pattern

Step 3 — Prepare the variables and split the data

Create X and y, then separate training and test sets

Step 4 — Fit the regression model

Estimate the slope and intercept from training data

Step 5 — Predict and interpret

Use the fitted line, understand its coefficients, and examine its limits

The notebook uses a dataset named `placement.csv`, with `cgpa` as the input variable and `package` as the target variable. The CSV itself is not included with the notebook you uploaded, so we can explain the exact code and mathematics, but we cannot responsibly claim numerical coefficients or actual predictions from your dataset yet.

## 2. First, understand the statistical problem

Imagine a dataset containing students' CGPA scores and their placement packages.

| Student | CGPA (input) | Package in LPA (target) |
| ------- | ------------ | ----------------------- |
| A       | 6.0          | 2.5                     |
| B       | 6.5          | 3.0                     |
| C       | 7.0          | 3.2                     |
| D       | 7.5          | 4.0                     |
| E       | 8.0          | 4.5                     |
| F       | 8.5          | 5.0                     |

These six rows are illustrative teaching data, not observations taken from your CSV.

The statistical question is:

Can we estimate a student's expected package from their CGPA using a straight-line relationship?

We call CGPA the independent variable or predictor, and package the dependent variable or response.

In mathematical notation:

- \\(x\\): CGPA.
- \\(y\\): observed package.
- \\(\hat y\\): package predicted by the model.
- \\(m\\): slope of the line.
- \\(b\\): intercept of the line.

Our model is

\\[ \boxed{\hat y=mx+b} \\]

This simple equation contains the entire prediction mechanism.

- The slope \\(m\\) determines how much the predicted package changes when CGPA increases by one point.
- The intercept \\(b\\) determines the line's predicted value when CGPA is zero.
- The predicted value \\(\hat y\\) is the model's estimate, not necessarily the student's actual package.

The central statistical idea: We do not expect every student to lie exactly on a straight line. We seek a line that represents the overall trend as well as possible.

## 3. Step 1 — Import the libraries

Your first notebook cell is:

```
import matplotlib.pyplot as pltimport pandas as pdimport numpy as np
```

Each library solves a different problem.

| Library             | Purpose                        | Statistical role                                       |
| ------------------- | ------------------------------ | ------------------------------------------------------ |
| `matplotlib.pyplot` | Draw graphs                    | Visualize relationships and fitted lines               |
| `pandas`            | Read and organize tabular data | Represent observations as rows and columns             |
| `numpy`             | Numerical computation          | Work with arrays, vectors, and mathematical operations |

For example:

```
import pandas as pddf = pd.read_csv("placement.csv")
```

Pandas turns a CSV file into a DataFrame, which is essentially a table of observations.

Why do we need this? Statistical learning begins with observed data. Before estimating any mathematical relationship, we need a collection of paired measurements.

Each row represents one observation:

\\[ (x_i,y_i) \\]

Here, \\(i\\) identifies the student. If the dataset has \\(n\\) students, it contains \\(n\\) pairs:

\\[ (x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n) \\]

The subscript matters. \\(x_1\\) is the first student's CGPA; \\(y_1\\) is that student's observed package. These are individual observations, not the model's slope or intercept.

## 4. Step 2 — Load and inspect the dataset

Your notebook runs:

```
df = pd.read_csv('placement.csv')df.head()
```

`pd.read_csv()` loads the observations, and `df.head()` displays the first five rows by default.

Why inspect the data before modeling?

Because a mathematical model is only as meaningful as the data it receives. You should verify:

- Are the columns named as expected?
- Are the values numeric?
- Are missing values present?
- Are there unusual or incorrect observations?
- Does each row represent the intended observational unit?

You can investigate with:

```
print(df.shape)print(df.columns)print(df.dtypes)print(df.isna().sum())print(df.describe())
```

### The statistical concepts behind inspection

Sample size, \\(n\\): The number of observations available. More observations can provide more information, but only if their quality and relevance are adequate.

Mean: The arithmetic average of a variable.

\\[ \bar x=\frac{1}{n}\sum\_{i=1}^{n}x_i \\]

Variance: A measure of how spread out the observations are around their mean.

\\[ s_x^2=\frac{\sum\_{i=1}^{n}(x_i-\bar x)^2}{n-1} \\]

The denominator \\(n-1\\) is used here for the usual unbiased sample variance estimate.

Standard deviation: The square root of variance, expressed in the original variable's units.

\\[ s_x=\sqrt{s_x^2} \\]

Why do these matter for regression? If CGPA barely varies between students, the dataset may provide little information about how package changes across different CGPAs. If the data contain major errors or outliers, the estimated line may be distorted.

## 5. Step 3 — Plot the data using a scatter plot

Your notebook contains:

```
plt.scatter(df['cgpa'], df['package'])plt.xlabel('CGPA')plt.ylabel('Package(in lpa)')
```

Each dot represents one student's observed CGPA and package.

### Illustrative scatter plot

Example data only — not your CSV.

2.25 LPA3 LPA3.75 LPA4.5 LPA5.25 LPA66.87.58.39

A positive pattern suggests that higher CGPA tends to accompany higher package in this illustrative sample.

### What does a scatter plot tell us statistically?

First, look at the direction of the relationship.

- Positive: larger \\(x\\) tends to accompany larger \\(y\\).
- Negative: larger \\(x\\) tends to accompany smaller \\(y\\).
- Little apparent linear association: no clear upward or downward straight-line pattern.

Second, inspect the form of the relationship. Does it look approximately linear, curved, or divided into different clusters?

Third, inspect the spread. Are the points tightly grouped around a potential line, or are they widely dispersed?

Finally, look for outliers — observations that lie unusually far from the general pattern.

### The connection to covariance and correlation

The sample covariance measures how two variables vary together:

\\[ s\_{xy}=\frac{\sum\_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}{n-1} \\]

When both variables tend to be above their respective means together, and below their means together, covariance tends to be positive.

Because covariance depends on the units of the variables, we often use the Pearson correlation coefficient:

\\[ r=\frac{s\_{xy}}{s_xs_y} \\]

Its value lies between \\(-1\\) and \\(+1\\).

- \\(r\\) near \\(+1\\): strong positive linear association.
- \\(r\\) near \\(-1\\): strong negative linear association.
- \\(r\\) near \\(0\\): little linear association.

Be careful: a correlation near zero does not rule out a nonlinear relationship. And even a strong correlation does not establish that CGPA causes a higher package. Other factors can influence placements.

## 6. Step 4 — Separate the predictor \\(X\\) and target \\(y\\)

Your notebook:

```
X = df.iloc[:, 0:1]y = df.iloc[:, -1]
```

This is one of the most important steps to understand.

- `X` selects the first column as the input.
- `y` selects the last column as the target.

In the intended dataset, the first column is CGPA and the last column is package.

There is a subtle Python detail here:

```
X = df.iloc[:, 0:1]
```

The slice `0:1` selects column zero while preserving a two-dimensional DataFrame. By contrast:

```
X = df.iloc[:, 0]
```

usually produces a one-dimensional Series.

Scikit-learn expects a feature matrix with shape:

\\[ X\in\mathbb R^{n\times p} \\]

where \\(n\\) is the number of observations and \\(p\\) is the number of predictor features.

For simple linear regression, \\(p=1\\). Thus, \\(X\\) has one column and \\(n\\) rows.

The target is a vector:

\\[ y\in\mathbb R^n \\]

Each position in \\(X\\) must correspond to the target in the same position in \\(y\\).

For example:

X

CGPA

Model

Learn line

ŷ

Predicted package

During training, the model also receives the true target y to learn the line. During prediction, it receives new X values.

### Why is `y` a Series?

Your next cell is:

```
y
```

In a Jupyter notebook, putting a variable name in the last line of a cell displays its value. This cell is for inspection; it does not train the model.

One important warning: selecting columns by position only works if the column order is correct. A more explicit alternative is:

```
X = df[['cgpa']]y = df['package']
```

This makes your intended predictor and target clear and reduces the chance of accidentally training on the wrong columns.

## 7. Step 5 — Split the data into training and test sets

Your notebook:

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=2)
```

This step protects you from a fundamental mistake: evaluating a model only on the same data it used to learn.

### What is training data?

Training data teach the model the relationship between CGPA and package.

The model sees the training predictors \\(X\_{\text{train}}\\) and their corresponding observed targets \\(y\_{\text{train}}\\). It uses them to estimate the slope and intercept.

### What is test data?

Test data are held back while the model is fitted. They provide an estimate of how well the fitted model predicts observations it did not train on.

For example, if there are 100 observations and `test_size=0.2`, the split ordinarily produces 80 training observations and 20 test observations.

### Visualize the split

80% training

20% testing

Illustration of the proportions. The actual number of rows depends on your dataset.

### Why `random_state=2`?

The split is randomized. `random_state=2` fixes the random seed so you can reproduce the same split when running the notebook with the same data and software behavior.

The number 2 has no special statistical meaning. It is not a model parameter and does not make the regression more accurate.

### Why does this matter statistically?

A model can fit training observations reasonably well yet perform poorly on new observations. This is called poor generalization.

The test set gives us a more honest assessment than evaluating only on training data, provided we do not repeatedly use the test set to make modeling decisions.

For very small datasets, a single split can produce an unstable evaluation. Cross-validation can help assess how sensitive the results are to the chosen split.

An important limitation in your notebook: It creates a test set but does not calculate an evaluation metric such as mean absolute error, mean squared error, or \\(R^2\\). We will see how to add that after understanding the line.

## 8. Step 6 — Import and create the regression model

Your next cells are:

```
from sklearn.linear_model import LinearRegressionlr = LinearRegression()
```

The first statement imports the estimator. The second creates a model object.

At this point, the model has not learned anything from your dataset. We have created the mechanism that will estimate the line.

The model is based on this mathematical relationship:

\\[ y_i=mx_i+b+\varepsilon_i \\]

Let's unpack it.

| Symbol              | Statistical meaning                                            |
| ------------------- | -------------------------------------------------------------- |
| \\(y_i\\)           | Actual observed package for student \\(i\\)                    |
| \\(x_i\\)           | Student \\(i\\)'s CGPA                                         |
| \\(m\\)             | Population or estimated regression slope, depending on context |
| \\(b\\)             | Population or estimated intercept, depending on context        |
| \\(\varepsilon_i\\) | Unexplained random deviation in the statistical model          |

For the fitted model, we usually write:

\\[ \hat y_i=\hat m x_i+\hat b \\]

The hats mean that these are estimates learned from the sample.

Why include \\(\varepsilon_i\\) in the statistical model? Because a student's package is influenced by more than CGPA. Skills, interview performance, job market conditions, company selection, and other factors may affect the result.

The line represents the systematic linear part of the relationship. The residuals describe how observations deviate from the fitted line.

## 9. Step 7 — Fit the model: the mathematics behind `lr.fit()`

Your notebook runs:

```
lr.fit(X_train, y_train)
```

This is the main learning step.

Scikit-learn estimates the line's parameters from the training observations. By default, `LinearRegression` includes an intercept and solves an ordinary least-squares problem.

### 9.1 What is a residual?

For each training observation, the model predicts a value:

\\[ \hat y_i=\hat m x_i+\hat b \\]

The residual is the difference between the actual value and the prediction:

\\[ e_i=y_i-\hat y_i \\]

Suppose a student has an observed package of 5 LPA, while the model predicts 4.4 LPA.

\\[ e_i=5-4.4=0.6 \\]

The model underpredicted that observation by 0.6 LPA.

If the model predicted 5.6 LPA instead:

\\[ e_i=5-5.6=-0.6 \\]

The residual is negative because the prediction was too high.

### A residual is a vertical distance

The orange dashed segment illustrates the difference between an observed point and its prediction on the blue line.

### 9.2 Why not just minimize the sum of residuals?

One might think that the best line should minimize:

\\[ \sum\_{i=1}^{n}e_i \\]

But positive and negative residuals can cancel each other out. A line could have large prediction errors yet a residual sum close to zero.

Instead, ordinary least squares minimizes the sum of squared residuals:

\\[ \boxed{ SSE(m,b)=\sum\_{i=1}^{n} \left[y_i-(mx_i+b)\right]^2 } \\]

SSE stands for sum of squared errors in this regression context, more precisely the sum of squared residuals.

Squaring does three things:

1. It prevents positive and negative residuals from canceling.
2. It penalizes large errors more heavily than small ones.
3. It creates a convenient, differentiable mathematical objective.

For example, residuals of \\(+2\\) and \\(-2\\) sum to zero, but their squared sum is:

\\[ 2^2+(-2)^2=8 \\]

The objective now reflects the size of the errors.

### 9.3 How does least squares choose the slope and intercept?

For observations \\((x_i,y_i)\\), the ordinary least-squares estimates, when the predictor has nonzero variance, are:

\\[ \boxed{ \hat m= \frac{\sum\_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)} {\sum\_{i=1}^{n}(x_i-\bar x)^2} } \\]

and

\\[ \boxed{\hat b=\bar y-\hat m\bar x} \\]

Here, \\(\bar x\\) and \\(\bar y\\) are the sample means.

Notice the structure of the slope formula:

- The numerator measures how \\(x\\) and \\(y\\) vary together.
- The denominator measures the variation in \\(x\\).
- Their ratio gives the slope of the least-squares line.

The intercept is then chosen so the fitted line passes through the point \\((\bar x,\bar y)\\).

That last fact is a useful geometric result of ordinary least squares when the model includes an intercept.

### 9.4 Why does this formula minimize squared error?

We can write the objective as:

\\[ SSE(m,b)=\sum_i[y_i-(mx_i+b)]^2 \\]

Differentiating with respect to \\(b\\) and setting the derivative to zero gives:

\\[ \sum_i(y_i-\hat m x_i-\hat b)=0 \\]

This says that the residuals sum to zero on the training sample when the model includes an intercept.

Differentiating with respect to \\(m\\) and setting the derivative to zero gives:

\\[ \sum_i x_i(y_i-\hat m x_i-\hat b)=0 \\]

These are the normal equations for simple linear regression. Solving them yields the slope and intercept formulas above.

In other words, the fitted coefficients are not arbitrary. They are the values that minimize the squared residual objective.

### 9.5 Where does gradient descent fit into this?

Gradient descent solves the same optimization problem through iterative updates rather than directly solving the least-squares equations.

For the mean squared error objective,

\\[ J(m,b)=\frac{1}{n}\sum\_{i=1}^{n} (mx_i+b-y_i)^2 \\]

the gradients are:

\\[ \frac{\partial J}{\partial m} =\frac{2}{n}\sum_i x_i(mx_i+b-y_i) \\]

\\[ \frac{\partial J}{\partial b} =\frac{2}{n}\sum_i(mx_i+b-y_i) \\]

The updates are:

\\[ m\leftarrow m-\alpha\frac{\partial J}{\partial m} \\]

\\[ b\leftarrow b-\alpha\frac{\partial J}{\partial b} \\]

Here, \\(\alpha\\) is the learning rate: it controls the size of each step in parameter space.

- If the learning rate is too large, updates may overshoot the minimum or diverge.
- If it is too small, convergence can be slow.
- For this convex quadratic objective, an appropriate learning rate can lead to the global minimum.

The key distinction is that your current notebook does not implement these updates. Scikit-learn's `LinearRegression` obtains the ordinary least-squares solution using numerical linear algebra. Gradient descent is another way to optimize the same objective, often useful when the problem is large or when you want to study iterative learning directly.

## 10. Step 8 — Inspect the test data

Your notebook contains:

```
X_test
```

and then:

```
y_test
```

These cells display the input values and actual target values held out during the split.

For example, a test observation might have:

\\[ x_i=8.0,\qquad y_i=4.5 \\]

Here, 8.0 is the student's CGPA and 4.5 LPA is the observed package. We can ask the fitted model to predict the package for CGPA 8.0 and compare that prediction with 4.5.

This comparison is essential because the actual package is known for the test observation, even though that observation was not used to fit the line.

The prediction error on the test set is:

\\[ e_i=y_i-\hat y_i \\]

A test residual measures how far the model's prediction is from the observed outcome for an unseen training example.

## 11. Step 9 — Predict one student's package

Your notebook:

```
lr.predict(X_test.iloc[0].values.reshape(1, 1))
```

Let's break down every operation.

`X_test.iloc[0]`

Selects the first test observation. Because `X_test` is a DataFrame, the selected row is a Series.

`.values`

Converts the selected row to a NumPy array of values.

`.reshape(1, 1)`

Makes the array two-dimensional with one observation and one feature. Its shape becomes `(1, 1)`.

`lr.predict(...)`

Uses the learned coefficients to calculate the predicted package.

The model calculates:

\\[ \hat y=\hat m x+\hat b \\]

For a purely illustrative example, suppose a fitted line has slope \\(1.2\\) and intercept \\(-4.0\\). At a CGPA of 8.0:

\\[ \hat y=1.2(8.0)-4.0=5.6 \\]

The predicted package would be 5.6 LPA. These coefficients are hypothetical and are not results from your notebook.

A cleaner prediction using the DataFrame is:

```
lr.predict(X_test.iloc[[0]])
```

Double brackets preserve the two-dimensional DataFrame shape and column name, making the code easier to read.

## 12. Step 10 — Draw the regression line

Your notebook:

```
plt.scatter(df['cgpa'], df['package'])plt.plot(    X_train,    lr.predict(X_train),    color='red')plt.xlabel('CGPA')plt.ylabel('Package(in lpa)')
```

This graph combines two things:

- The scatter plot shows observed data.
- The red line shows the model's predicted values over the training inputs.

The line is not supposed to pass through every dot. Instead, it represents the fitted linear trend.

For a given CGPA, the vertical distance between a dot and the line is the residual for that observation.

One technical detail: the line is plotted using `X_train`, which may not be sorted. If the training CGPA values are not ordered, Matplotlib can connect the predicted points in an irregular sequence, making the line appear jagged.

You can make the line cleaner:

```
x_line = pd.DataFrame({    'cgpa': np.linspace(        X_train['cgpa'].min(),        X_train['cgpa'].max(),        100    )})y_line = lr.predict(x_line)plt.scatter(df['cgpa'], df['package'])plt.plot(x_line['cgpa'], y_line, color='red')plt.xlabel('CGPA')plt.ylabel('Package (LPA)')plt.show()
```

This assumes the predictor column is named `cgpa`, as in your notebook.

`np.linspace()` creates 100 evenly spaced CGPA values. The model predicts a package for each value, and the sorted inputs allow Matplotlib to draw a continuous-looking straight line.

## 13. Step 11 — Extract the slope

Your notebook:

```
m = lr.coef_m
```

For simple linear regression with one feature, `lr.coef_` contains the estimated slope.

Mathematically:

\\[ \hat m= \frac{\sum_i(x_i-\bar x)(y_i-\bar y)} {\sum_i(x_i-\bar x)^2} \\]

### What does the slope mean?

Suppose the fitted slope were \\(1.2\\).

That would mean the model predicts an increase of 1.2 LPA for each one-point increase in CGPA, on average according to the fitted linear relationship.

It does not mean every student who improves their CGPA by one point will receive a package increase of exactly 1.2 LPA.

The units of the slope are:

\\[ \frac{\text{LPA}}{\text{CGPA point}} \\]

A positive slope indicates an upward fitted relationship; a negative slope indicates a downward one.

## 14. Step 12 — Extract the intercept

Your notebook:

```
b = lr.intercept_b
```

The intercept is the fitted line's value when \\(x=0\\):

\\[ \hat y=\hat m(0)+\hat b=\hat b \\]

It also satisfies:

\\[ \boxed{\hat b=\bar y-\hat m\bar x} \\]

This is how the estimated slope and intercept relate to the sample means.

However, interpret the intercept carefully. If the dataset only contains CGPAs from, say, 5 to 10, a CGPA of zero is outside the observed range. The intercept may then have little practical meaning, even though it is mathematically necessary to define the fitted line.

A negative intercept is not automatically an error. It can simply be where the mathematical line crosses the vertical axis.

## 15. Steps 13–15 — Use the equation to make predictions

Your notebook ends with:

```
# y = mx + bm * 8.58 + b
```

Then:

```
m * 9.5 + b
```

And finally:

```
m * 100 + b
```

The first two expressions substitute CGPA values into the fitted equation. The last expression substitutes 100, which illustrates an important concept: a model will return a number even when the input is unreasonable for the data.

There is a small implementation detail to know: Scikit-learn's `lr.coef_` is normally a one-element NumPy array for this single-feature model. Consequently, `m * 8.58 + b` may display a one-element array rather than a scalar.

You can use:

```
slope = float(lr.coef_[0])intercept = float(lr.intercept_)prediction_858 = slope * 8.58 + interceptprediction_95 = slope * 9.5 + interceptprediction_100 = slope * 100 + intercept
```

The mathematics is identical.

### Why is predicting at CGPA 100 problematic?

A CGPA of 100 is far outside the typical CGPA scale. The model still calculates a result because its equation accepts numeric inputs, but that does not make the prediction meaningful.

This is called extrapolation: predicting outside the range of observed predictor values.

Predictions are generally more defensible within the range of the training data, assuming the linear relationship is reasonable there. Even within that range, the model can be wrong.

A useful rule:

- Interpolation: Predicting within the observed range of CGPA values.
- Extrapolation: Predicting beyond that range.

Do not confuse the ability to compute a number with evidence that the number is accurate.

## 16. What the notebook still needs: evaluate the model statistically

Your notebook builds the line and makes predictions, but it does not yet quantify how accurate those predictions are.

Add this after fitting the model:

```
from sklearn.metrics import (    mean_absolute_error,    mean_squared_error,    r2_score)y_pred = lr.predict(X_test)mae = mean_absolute_error(y_test, y_pred)mse = mean_squared_error(y_test, y_pred)rmse = np.sqrt(mse)r2 = r2_score(y_test, y_pred)print("MAE:", mae)print("MSE:", mse)print("RMSE:", rmse)print("R²:", r2)
```

These calculations use the held-out test set.

### Mean absolute error (MAE)

\\[ MAE=\frac{1}{n}\sum_i|y_i-\hat y_i| \\]

MAE calculates the average absolute prediction error. If MAE is 0.5, the model's test predictions differ from actual packages by an average of 0.5 LPA in absolute terms.

### Mean squared error (MSE)

\\[ MSE=\frac{1}{n}\sum_i(y_i-\hat y_i)^2 \\]

MSE squares the errors, so larger mistakes contribute disproportionately. Its units are LPA squared.

### Root mean squared error (RMSE)

\\[ RMSE=\sqrt{MSE} \\]

RMSE returns the error to the original target units, LPA. Like MSE, it penalizes larger errors more strongly than MAE.

### Coefficient of determination (\\(R^2\\))

\\[ R^2=1-\frac{\sum_i(y_i-\hat y_i)^2} {\sum_i(y_i-\bar y)^2} \\]

The denominator measures the test targets' total squared deviation from their test-set mean.

An \\(R^2\\) of 0 means the model's squared-error performance equals that of predicting the test-set mean, using this metric's definition. An \\(R^2\\) of 1 indicates perfect predictions. A negative test \\(R^2\\) is possible and means the model performs worse in squared-error terms than that mean-prediction baseline on the test set.

A high \\(R^2\\) alone does not prove the model is suitable. Examine error size, the data distribution, residual patterns, and whether the test set represents the population you want to predict.

## 17. The statistical assumptions behind simple linear regression

To understand regression deeply, you need to distinguish the computational objective from the assumptions used for statistical inference.

Ordinary least squares can calculate a fitted line without all the following assumptions being true. But assumptions matter when interpreting the model and making statistical claims.

| Assumption or condition                            | Meaning                                                                   | Why it matters                                                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Linear conditional relationship                    | The expected value of the target is approximately linear in the predictor | A straight line can be systematically wrong for a curved relationship                                   |
| Independent errors                                 | Errors are not systematically dependent across observations               | Dependence can invalidate conventional uncertainty estimates                                            |
| Constant error variance                            | Error spread is roughly constant across predictor values                  | Unequal variance affects conventional standard errors and inference                                     |
| Errors have mean zero conditional on the predictor | No systematic unexplained shift remains at each predictor value           | Supports interpretation of the conditional mean                                                         |
| Normally distributed errors                        | Errors are approximately normal conditional on the predictors             | Useful for exact small-sample inference under the standard model, but not required just to fit the line |

For causal interpretation, additional assumptions are needed. A positive regression slope alone does not show that changing CGPA causes package changes.

For example, if CGPA is associated with interview preparation, and interview preparation also affects package, then the observed relationship may partly reflect that other factor.

## 18. Practice: test your statistical understanding

Try these questions before moving to multiple linear regression or implementing gradient descent.

1\. What does the slope m represent?

The predicted package when CGPA is zero

The predicted change in package for a one-point increase in CGPA

The number of students in the dataset

2\. Why does ordinary least squares square residuals?

To make all residuals positive and penalize larger errors more

To guarantee every point lies on the line

To make the intercept equal to zero

3\. What does random_state=2 do?

Sets the slope to 2

Makes predictions two times larger

Makes the randomized split reproducible

4\. Why keep test observations out of fitting?

So the model has fewer columns

To evaluate predictions on observations not used to fit the coefficients

To force R² to be positive

5\. What is a problem with predicting at CGPA 100?

The equation cannot accept numbers greater than 10

It is likely extreme extrapolation beyond the observed CGPA range

The slope automatically becomes zero

Reset quiz

## 19. Your next practical step

Upload the `placement.csv` file that this notebook expects. Then we can use your actual observations to calculate the sample means, covariance, variance, slope, intercept, residuals, and least-squares objective step by step, checking each statistical calculation against Scikit-learn's fitted results.

After that, the natural next lesson is to implement gradient descent from scratch on the same dataset, visualize how the slope and intercept change during training, and compare its final coefficients with `LinearRegression`. That will connect the statistics in this notebook to the iterative optimization used in many machine-learning algorithms.