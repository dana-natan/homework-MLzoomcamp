# Machine Learning Workflow: Step-by-Step Reference

## First: Understanding the variable names

There are three main groups of variables in our homework.

**1. DataFrames (`df`): The original data**

| Variable | Meaning |
|---|---|
| `df` | The full dataset |
| `df_train` | Training observations, 6,000 cars |
| `df_val` | Validation observations, 2,000 cars |
| `df_test` | Test observations, 2,000 cars |

`df_train`, `df_val`, and `df_test` still contain both the predictors and the target.

**2. Features (`X`) and targets (`y`)**

| Variable | Meaning |
|---|---|
| `X_train` | Training input features |
| `y_train` | Actual MPG values for training cars |
| `X_val` | Validation input features |
| `y_val` | Actual MPG values for validation cars |
| `X_test` | Test input features |
| `y_test` | Actual MPG values for test cars |

**Remember:**

- `X` = what we know about each car (horsepower, weight, engine, year).
- `y` = what we want to predict (MPG).
- `train`, `val`, `test` = which portion of the dataset we're using.

**3. Prepared features, learned weights, and predictions**

| Variable | Meaning |
|---|---|
| `X_train_zero` | Training features with missing values filled with 0 |
| `X_val_zero` | Validation features with missing values filled with 0 |
| `X_train_mean` | Training features with mean imputation |
| `X_val_mean` | Validation features using the training mean |
| `w0` | Learned intercept |
| `w` | Learned feature coefficients |
| `r` | Regularization strength |
| `y_pred` | MPG values predicted by the model |
| `rmse` | Prediction error summary |

`X_train_zero` and `X_train_mean` are two different prepared versions of the same training features. We compare them because Homework 2 asks which imputation approach gives better predictions.

---

## Step 1: EXPLORE. What data do I have?

Before modeling, inspect the data, identify the target, check for missing values, and examine distributions.

```python
df.head()
df.shape
df.describe()
df.isna().sum()
```

**Goal:** Understand the data before deciding how to model it.

## Step 2: SPLIT. Which observations can I learn from?

We divide the dataset into three parts:

```python
df_train   # 60%: learn
df_val     # 20%: compare decisions
df_test    # 20%: final evaluation
```

In our homework, these are created by shuffling the observations with a fixed random seed and selecting the appropriate rows.

**Goal:** Keep validation and test observations separate from the data used to learn model parameters.

## Step 3: PREPARE. What are X and y?

Separate the predictor columns from the outcome column.

```python
features = [
    'engine_displacement',
    'horsepower',
    'vehicle_weight',
    'model_year'
]

X_train = df_train[features]
y_train = df_train['fuel_efficiency_mpg']

X_val = df_val[features]
y_val = df_val['fuel_efficiency_mpg']
```

Then handle missing values. For zero imputation:

```python
X_train_zero = X_train.fillna(0)
X_val_zero = X_val.fillna(0)
```

For mean imputation, calculate the mean from training only:

```python
horsepower_mean = X_train['horsepower'].mean()

X_train_mean = X_train.fillna({'horsepower': horsepower_mean})
X_val_mean = X_val.fillna({'horsepower': horsepower_mean})
```

**Goal:** Produce usable input matrices while avoiding data leakage.

## Step 4: TRAIN. Learn the weights from X_train and y_train.

The model uses the training features and actual target values to learn coefficients.

For Question 4, we use our regularized function:

```python
w0, w = train_linear_regression_reg(
    X_train_zero.to_numpy(),
    y_train,
    r
)
```

What goes in:

- `X_train_zero`: the training features.
- `y_train`: the correct MPG values.
- `r`: the regularization strength.

What comes out:

- `w0`: the intercept.
- `w`: one coefficient for each feature.

### Where do X.T and XTX belong?

Inside the training function:

```python
XTX = X.T @ X
```

- `X` is the training feature matrix passed into the function.
- `.T` means transpose, swapping rows and columns.
- `@` means matrix multiplication.
- `XTX` is an intermediate matrix used to calculate regression coefficients.

In our function, an intercept column is added first, so the training matrix has five columns.

We add regularization with:

```python
XTX = XTX + r * np.eye(XTX.shape[0])
```

Then the function calculates the coefficients.

**Important:** `XTX` is not a dataset and is not a prediction. It's an intermediate mathematical calculation used during training.

## Step 5: PREDICT. Apply the learned weights to validation cars.

Once we've trained the model, we use its coefficients to predict MPG for validation observations.

```python
y_pred = w0 + X_val_zero.to_numpy() @ w
```

Read this as:

**Predicted MPG = intercept + validation features multiplied by learned coefficients.**

- `X_val_zero`: the validation features.
- `w`: coefficients learned during training.
- `w0`: the learned intercept.
- `y_pred`: predicted MPG for each validation car.

No new coefficients are learned in this step.

**Important difference:**

- `X.T @ X` is used inside our training function to learn weights.
- `X_val_zero.to_numpy() @ w` applies the learned weights to generate predictions.

## Step 6: EVALUATE. How accurate are the predictions?

Compare predicted MPG with actual validation MPG.

```python
mse = np.mean((y_val.to_numpy() - y_pred) ** 2)
rmse = np.sqrt(mse)
```

- `y_val`: actual MPG.
- `y_pred`: predicted MPG.
- Subtract to get the prediction errors.
- Square the errors, average them, and take the square root.

**Lower RMSE means better predictive performance on that validation dataset.**

## Step 7: COMPARE. Which modeling settings work best?

For Question 4, repeat training, prediction, and evaluation for these regularization values:

```python
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
```

For every `r`:

1. Train a model with `X_train_zero` and `y_train`.
2. Generate `y_pred` using `X_val_zero`.
3. Calculate validation RMSE.
4. Record the RMSE and compare the results.

Choose the `r` with the lowest validation RMSE, rounded according to the homework instructions. If there's a tie, select the smallest `r`.

## Step 8: FINAL TEST. How well does the selected model generalize?

After choosing the model settings:

1. Combine the training and validation observations.
2. Prepare the combined data using the selected preprocessing rules.
3. Train a new model on the combined data.
4. Generate predictions for `X_test`.
5. Compare those predictions with `y_test`.

For Homework 2, Question 6 specifically requires seed 9, zero imputation, and `r = 0.001`.

The test set has remained untouched during model selection, providing a more independent estimate of performance.

---

## Quick memory reference

| Name or expression | Think of it as |
|---|---|
| `df` | Dataset |
| `X` | Questions/input information |
| `y` | Correct answers/target values |
| `_train` | Data used to learn |
| `_val` | Data used to compare models |
| `_test` | Data reserved for final evaluation |
| `_zero` | Missing values replaced by 0 |
| `_mean` | Missing values replaced by training mean |
| `.T` | Transpose a matrix |
| `@` | Matrix multiplication |
| `X.T @ X` | Part of coefficient estimation |
| `w0`, `w` | Learned intercept and weights |
| `y_pred` | Model's predicted answers |
| `RMSE` | How far predictions are from actual values |

**The entire process in one sentence:**

Explore the data → split it → prepare X and y → train to learn weights → predict using those weights → compare predictions against actual y → choose settings on validation → evaluate the final model on test.


**The most important distinction for you to remember is this:**

X.T @ X belongs inside our function for learning the weights. 
X_val @ w belongs to the prediction stage, where we reuse those weights. 

Both use matrix multiplication, but they're doing different jobs.

-----

## Python Cheat Sheet: Libraries and Functions

A reference for the Python tools used throughout ML Zoomcamp. You don't need to memorize every command, but you should understand what each one does and when to use it.

### 1. Main libraries

| Library | Import | What it's for |
|---|---|---|
| **Pandas** | `import pandas as pd` | Loading, exploring, cleaning, and manipulating tabular data |
| **NumPy** | `import numpy as np` | Arrays, mathematical operations, matrices, and linear algebra |
| **Matplotlib** | `import matplotlib.pyplot as plt` | Creating plots and visualizations |
| **Seaborn** | `import seaborn as sns` | Statistical visualizations, built on Matplotlib |
| **scikit-learn** | `from sklearn.linear_model import LinearRegression` | Training, evaluating, and preprocessing machine learning models |

### 2. Pandas: Working with datasets

Pandas is usually our starting point when working with tabular data.

| Command | What it does |
|---|---|
| `pd.read_csv(url)` | Load a CSV file into a DataFrame |
| `df.head()` | Show the first 5 rows |
| `df.shape` | Show the number of rows and columns |
| `df.columns` | List column names |
| `df.dtypes` | Show each column's data type |
| `df.info()` | Summarize columns, types, and non-missing counts |
| `df.describe()` | Show descriptive statistics for numerical columns |
| `df.isna().sum()` | Count missing values in each column |
| `df['horsepower'].mean()` | Calculate the mean |
| `df['horsepower'].median()` | Calculate the median |
| `df['horsepower'].fillna(0)` | Replace missing values with zero |
| `df.copy()` | Create a separate copy of a DataFrame |
| `df[features]` | Select multiple columns from a list |
| `df.iloc[0:10]` | Select rows by numerical position |
| `df.values` | Convert a DataFrame into a NumPy array |

**Important distinction:**

```python
df['horsepower']      # One column: Series
df[['horsepower']]    # One-column DataFrame
df[features]          # Multiple columns using a list
```

### 3. NumPy: Arrays and mathematical operations

NumPy is especially useful when implementing machine learning algorithms mathematically.

| Command | What it does |
|---|---|
| `np.array([1, 2, 3])` | Create an array |
| `np.arange(10)` | Create numbers from 0 to 9 |
| `np.ones(5)` | Create five ones |
| `np.zeros(5)` | Create five zeros |
| `np.mean(x)` | Calculate the mean |
| `np.std(x)` | Calculate the standard deviation |
| `np.sqrt(x)` | Calculate the square root |
| `np.random.seed(42)` | Set the random seed for reproducibility |
| `np.random.shuffle(idx)` | Shuffle an array in place |
| `np.column_stack([a, b])` | Combine compatible arrays as columns |
| `X.T` | Transpose a matrix |
| `X @ w` | Matrix multiplication |
| `np.linalg.inv(X)` | Calculate a matrix inverse |
| `np.linalg.solve(A, b)` | Solve a linear system |

### 4. Python syntax we keep using

These are general Python operations, not functions from Pandas or NumPy.

| Syntax | Meaning |
|---|---|
| `len(df)` | Number of rows in a DataFrame |
| `print(x)` | Display a result |
| `round(x, 3)` | Round to 3 decimal places |
| `x = 5` | Assign a value to a variable |
| `def function_name():` | Define a function |
| `return x` | Return a result from a function |
| `for seed in seeds:` | Repeat an operation for each value |
| `x[0]` | First element |
| `x[1:]` | All elements except the first |
| `x[:5]` | First five elements |
| `X.shape[0]` | Number of rows in an array |
| `X.shape[1]` | Number of columns in an array |

Remember: Python indexing starts at **0**, not 1.

### 5. Visualizing data

We visualize data to understand distributions, detect outliers, and explore relationships before modeling.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['fuel_efficiency_mpg'], bins=30)
plt.show()
```

Frequently used functions:

| Function | Purpose |
|---|---|
| `sns.histplot()` | Display a distribution |
| `sns.scatterplot()` | Explore the relationship between two variables |
| `sns.boxplot()` | Examine spread and potential outliers |
| `plt.xlabel()` | Label the x-axis |
| `plt.ylabel()` | Label the y-axis |
| `plt.title()` | Add a plot title |
| `plt.show()` | Display the plot |

### 6. scikit-learn: Machine learning

In our homework, we implement linear regression ourselves to understand the mathematics. In professional projects, we can use an existing implementation.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

| Method | Meaning |
|---|---|
| `LinearRegression()` | Create a linear regression model |
| `model.fit(X, y)` | Train the model |
| `model.predict(X)` | Generate predictions |
| `model.coef_` | Access the learned feature coefficients |
| `model.intercept_` | Access the learned intercept |

The model inputs must contain valid numerical values, so missing-value handling is still necessary.

### 7. Common workflow to remember

Most introductory supervised machine learning projects follow this general structure:

1. **Load data:** `pd.read_csv()`
2. **Explore:** `.head()`, `.describe()`, `.isna().sum()`
3. **Prepare features and target:** `X`, `y`
4. **Split data:** Training, validation, test
5. **Preprocess:** Handle missing values and prepare variables
6. **Train:** Learn model parameters from training data
7. **Predict:** Apply the model to validation observations
8. **Evaluate:** Calculate an appropriate error metric
9. **Compare:** Choose preprocessing and model settings using validation
10. **Test:** Evaluate the final model on untouched test data

**What to remember:** The workflow and reasoning matter more than memorizing every function. Use this reference when you know what you want to do but can't recall the Python syntax.

---------

# ML Zoomcamp: Linear Regression Notes

Personal notes from Homework 1 and Homework 2.

## 1. What does linear regression do?

Linear regression predicts a numerical outcome (`y`) using one or more input variables (`X`).

For example, we want to predict a car's fuel efficiency (MPG) using horsepower, vehicle weight, engine displacement, and model year.

The model learns:
- **`w0` (intercept):** The baseline value in the prediction equation when all features equal zero.
- **`w` (weights):** The coefficients representing how much each feature contributes to the prediction, holding the other features constant.

The prediction is calculated as:

`prediction = w0 + w1*x1 + w2*x2 + ...`

The model learns the weights from training data and uses them to make predictions for new observations.

## 2. Training a linear regression model

We implemented linear regression ourselves using NumPy.

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)

    w = XTX_inv @ X.T @ y

    return w[0], w[1:]
```

### Understanding the code

**Step 1: Add the intercept column**

```python
ones = np.ones(X.shape[0])
X = np.column_stack([ones, X])
```

- `X.shape[0]` gives the number of rows (observations).
- `np.ones()` creates an array of ones, one per observation.
- `np.column_stack()` adds those ones as the first column of `X`.
- This allows the model to estimate the intercept along with the other coefficients.

**Step 2: Calculate the regression weights**

```python
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
w = XTX_inv @ X.T @ y
```

- `.T` transposes a matrix (swaps rows and columns).
- `@` performs matrix multiplication.
- `np.linalg.inv()` calculates the inverse of a matrix.
- These operations implement the ordinary least squares solution, finding the coefficients that minimize the sum of squared prediction errors.

The inverse exists only when `XTX` is invertible. This method is useful for learning the mathematics, but professional implementations generally use more numerically stable methods.

**Step 3: Return the coefficients**

```python
return w[0], w[1:]
```

The function returns two things:
- `w0`: The intercept, a single number.
- `w`: An array of weights, one for each input feature.

## 3. Training, validation, and test datasets

We divide the data into three sets:

| Dataset | Purpose |
|---|---|
| Training (60%) | Learn the model's coefficients |
| Validation (20%) | Compare modeling decisions, such as imputation or regularization |
| Test (20%) | Evaluate the final selected model on untouched data |

The test set should not influence model selection.

## 4. Handling missing values

In Homework 2, 877 out of 10,000 cars (8.77%) have missing horsepower values.

We compare two imputation strategies:

- **Zero imputation:** Replace missing values with `0`. This may introduce unrealistic values, but can sometimes produce better predictions.
- **Mean imputation:** Replace missing values with the average observed horsepower. This may be more plausible but reduces variability and can distort relationships.

Neither method is necessarily best. We compare validation RMSE to evaluate predictive performance.

**Important:** Calculate the mean using only the training data to avoid data leakage.

## 5. Preparing the features and target

```python
features = [
    'engine_displacement',
    'horsepower',
    'vehicle_weight',
    'model_year'
]

X_train = df_train[features]
y_train = df_train['fuel_efficiency_mpg']
```

- `X_train` contains the input features.
- `y_train` contains the actual values we want to predict.
- The target must not be included in the input features.

### Zero imputation

```python
X_train_zero = X_train.fillna(0)
X_val_zero = df_val[features].fillna(0)
```

Both sets receive the same preprocessing rule.

## 6. Training the model

```python
w0, w = train_linear_regression(
    X_train_zero.values,
    y_train.values
)
```

- `.values` converts a Pandas DataFrame or Series into a NumPy array.
- The function calculates the intercept and weights using the training observations.

## 7. Making predictions

Once we have the weights, we apply them to the validation features.

```python
y_pred_zero = w0 + X_val_zero.values @ w
```

Breaking it down:

- `X_val_zero.values` contains the feature values of all validation cars as a matrix.
- `@ w` multiplies each feature by its corresponding coefficient and sums the results for each car.
- `w0 +` adds the intercept to each prediction.
- `y_pred_zero` contains the predicted MPG for all validation cars.

**Key idea:** Training means learning the weights. Prediction means applying those learned weights to new data.

## 8. Evaluating predictions with RMSE

*To be completed as we continue Homework 2.*

## 9. Regularization

*To be completed in Homework 2, Question 4.*

## 10. Random seeds and model stability

*To be completed in Homework 2, Question 5.*

## 11. Final evaluation on the test set

*To be completed in Homework 2, Question 6.*