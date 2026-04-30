# Homework 1 - Logistic Regression on Real-World Data
### Binary Customer Classification | E-Commerce Dataset

---

## Context

You are working as an ML engineer at an e-commerce company.  
The business goal: **identify customers who are likely to stop engaging (churn/at-risk)
vs. customers who are healthy (loyal/VIP)** - a binary retention signal.

You will use the `LogisticRegression` class you implemented in labs 1 & 2,
and build a complete preprocessing + training + evaluation pipeline around it.

Dataset file: `ecommerce_customers.csv`

---

## Part 1 - Data Loading & Exploration

**Task 1.1** - Load the dataset. Print its shape, dtypes, and the first 5 rows.

**Task 1.2** - Inspect the target column `customer_segment`.
Identify the four class values and their counts.
Create a new binary column `target`:
  - `0` → customer is **disengaged** (Churned or At-Risk)
  - `1` → customer is **healthy** (Loyal or VIP)

Print the class balance of the new `target` column.
Is the dataset balanced enough to use raw accuracy as a metric? Justify your answer.

**Task 1.3** - Perform an exploratory analysis:
  - How many rows contain at least one missing or corrupted value?
  - For each numerical column, report the number of non-numeric entries (string
    sentinels such as "N/A", "error", etc.).
  - Plot a histogram or boxplot for at least 3 numerical features, coloured by
    the binary `target`. What patterns do you see?

---

## Part 2 - Preprocessing Pipeline

You must handle: missing values, corrupt non-numeric entries, outliers,
categorical encoding, and feature scaling - all inside an `sklearn` Pipeline.

**Task 2.1** - Write a custom `sklearn`-compatible transformer class
`NumericCleaner` (inheriting from `BaseEstimator, TransformerMixin`) that:
  - Coerces each column to numeric with `pd.to_numeric(..., errors='coerce')`,
    turning string sentinels into `NaN`.
  - Clips values to a valid range per column (e.g. age ∈ [18, 100],
    total_spent ≥ 0, discount_usage_rate ∈ [0, 1]).
  - Returns a cleaned numpy array.

**Task 2.2** - Build a full `ColumnTransformer` + `Pipeline`:
  - **Numerical branch**: `NumericCleaner` → `SimpleImputer(strategy='median')` →
    `StandardScaler`
  - **Categorical branch**: `SimpleImputer(strategy='most_frequent')` →
    `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

Drop the `customer_id` column before feeding data into the pipeline.

**Task 2.3** - Fit the pipeline on the training set only. Transform both train
and test sets. Report the shape of the resulting feature matrix.

**Task 2.4** - Split the dataset into train/test **before** fitting the pipeline.
Use an 80/20 stratified split (`stratify=target`, `random_state=42`).
Explain in 1–2 sentences why stratification matters here and why the pipeline
must be fit on the training set only.

---

## Part 3 - Training with Your LogisticRegression

Import your `LogisticRegression` class from `module_student.py` (or
`module_solution.py` if your implementation is not yet complete).

**Task 3.1** - Train a baseline model with default hyperparameters:
```
LogisticRegression(optimizer='mbgd', penalty='l2', learning_rate=0.1,
                   n_iterations=500, random_state=42)
```
Fit it on the preprocessed training data.

**Task 3.2** - Plot the training loss curve (and validation loss if early
stopping is enabled). Does the model converge? Does it overfit?

**Task 3.3** - Evaluate on the test set. Report:
  - Accuracy
  - Precision, Recall, F1-score for both classes (use `classification_report`)
  - Confusion matrix (plot as a heatmap)

Which metric matters most for the business goal described above?
Justify your choice in 2–3 sentences.

---

## Part 4 - Hyperparameter Search

**Task 4.1** - Use `sklearn.model_selection.RandomizedSearchCV` to search over
the following hyperparameter space (≥ 30 iterations, 5-fold CV, scoring='f1'):

```python
param_distributions = {
    "learning_rate":  [0.001, 0.01, 0.05, 0.1, 0.3],
    "n_iterations":   [200, 500, 1000],
    "optimizer":      ["bgd", "sgd", "mbgd"],
    "penalty":        ["l2", "l1", None],
    "lambda_":        [0.0001, 0.001, 0.01, 0.1, 1.0],
    "batch_size":     [16, 32, 64],          # only relevant for mbgd
}
```

Note: to use your custom class inside `RandomizedSearchCV`, it must implement
the `sklearn` estimator interface (`fit`, `predict`, `get_params`, `set_params`).
If your class does not yet have `get_params` / `set_params`, add them or wrap
the class in a thin `sklearn`-compatible wrapper.

**Task 4.2** - Print the best parameter combination and its CV F1 score.
Re-train the best model on the full training set and evaluate on the test set.
How much did the F1 score improve compared to the baseline?

**Task 4.3** - Plot a bar chart of the top 10 hyperparameter configurations
ranked by mean CV F1 score. Use `cv_results_` from the search object.

---

## Part 5 - Analysis & Visualisation

**Task 5.1** - Plot the weights of the best model (one bar per feature, sorted
by absolute magnitude). Which features are most informative for predicting
customer health? Does this match your intuition from the EDA in Part 1?

**Task 5.2** - Plot the Precision-Recall curve for the best model on the test
set. Choose and justify a decision threshold other than 0.5 (e.g. prioritising
recall to catch as many at-risk customers as possible).

**Task 5.3 (Optional / Bonus)** - Visualise the decision boundary:
  - Apply PCA (2 components) to the preprocessed test set.
  - Train a fresh `LogisticRegression` on the 2D PCA representation.
  - Plot the decision boundary as a filled contour, with test points overlaid
    and coloured by true label.
  - Add a short note on what information is lost by reducing to 2 dimensions.

---

## Deliverables

- Completed Jupyter notebook with all tasks answered.
- All plots embedded inline with axis labels and titles.
- Short written answers where requested (1–5 sentences each).
- `module_student.py` with your implementations.