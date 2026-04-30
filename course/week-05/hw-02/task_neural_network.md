# Homework 2 - Multi-Class Classification with a Neural Network
### Customer Segment Prediction | E-Commerce Dataset

---

## Context

In this homework you put the `NeuralNetwork` class you implemented in Lab 4 to
work on a real dataset.

The business goal: **predict which of the four customer segments a user belongs
to** - Churned, At-Risk, Loyal, or VIP - so the marketing team can apply
targeted retention and engagement strategies for each group.

Dataset file: `ecommerce_customers.csv` (same as in Homework 1)

---

## Part 1 - Data Loading & Exploration

**Task 1.1** - Load the dataset. The target column is `customer_segment` with
four classes. Print the class distribution as both counts and percentages.
Is the dataset balanced? How might imbalance affect a model trained with
categorical cross-entropy loss?

**Task 1.2** - Briefly compare the four classes across at least 3 numerical
features using boxplots or violin plots. Which features look most
discriminative? Do the patterns match the binary separation you observed in
Homework 1?

**Task 1.3** - Recall that the dataset contains corrupted cells (string
sentinels, impossible values, nulls). List the types of corruption present and
briefly describe how you plan to handle each one in the preprocessing pipeline.

---

## Part 2 - Preprocessing Pipeline

You may reuse or adapt the pipeline you built in Homework 1.

**Task 2.1** - Reuse the `NumericCleaner` transformer from Homework 1 to
coerce and clip numerical columns before imputation.

**Task 2.2** - Build a full `ColumnTransformer` + `Pipeline`:
  - **Numerical branch**: `NumericCleaner` → `SimpleImputer(strategy='median')` →
    `StandardScaler`
  - **Categorical branch**: `SimpleImputer(strategy='most_frequent')` →
    `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

Drop `customer_id` before fitting. Fit the pipeline on the training set only.

**Task 2.3** - Split the data 80/20, stratified by `customer_segment`,
`random_state=42`. Report the class distribution in both splits to confirm
stratification worked.

**Task 2.4** - Encode the target `customer_segment` into integer labels and a
one-hot matrix. Use `categories=[['Churned', 'At-Risk', 'Loyal', 'VIP']]` to
fix the column order:

```
Churned → 0,  At-Risk → 1,  Loyal → 2,  VIP → 3
```

Produce both representations and verify their consistency:

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# Integer labels - passed to NeuralNetwork.fit() and .predict()
le = LabelEncoder()
le.fit(['Churned', 'At-Risk', 'Loyal', 'VIP'])
y_train = le.transform(y_train_raw)   # shape (N,)
y_test  = le.transform(y_test_raw)

# One-hot matrix - used for manual inspection and optional tasks
ohe = OneHotEncoder(
    categories=[['Churned', 'At-Risk', 'Loyal', 'VIP']],
    sparse_output=False
)
Y_train_onehot = ohe.fit_transform(y_train_raw.reshape(-1, 1))  # shape (N, 4)
Y_test_onehot  = ohe.transform(y_test_raw.reshape(-1, 1))
```

Verify: `Y_train_onehot.shape` should be `(n_train_samples, 4)` and each row
must sum to 1. Confirm that `np.argmax(Y_train_onehot, axis=1)` equals
`y_train` exactly.

Note: `NeuralNetwork.fit()` accepts integer `y` of shape `(N,)` and builds
the one-hot encoding internally inside `compute_loss` and `backward`.
The `Y_*_onehot` arrays you produce here are for your own inspection only.

---

## Part 3 - Training

Import your `NeuralNetwork` class from `nn_multiclass_student.py`.

**Task 3.1** - Construct and train a baseline network:

```python
n_features = X_train.shape[1]

nn = NeuralNetwork(
    layer_sizes  = [n_features, 64, 32, 4],
    activations  = ['relu', 'relu', 'softmax'],
    random_state = 42
)
nn.fit(X_train, y_train, lr=0.01, n_epochs=500)
```

**Task 3.2** - Plot `nn.loss_curve_`. Does the loss decrease monotonically?
If you observe spikes or divergence, adjust `lr` or `n_epochs` and briefly
note what you changed and why.

---

## Part 4 - Evaluation

**Task 4.1** - Evaluate the baseline model on the test set. Report:
  - Overall accuracy
  - Per-class Precision, Recall, F1-score using:
    ```python
    classification_report(y_test, y_pred,
                          target_names=['Churned', 'At-Risk', 'Loyal', 'VIP'])
    ```
  - Macro-averaged and weighted-averaged F1

**Task 4.2** - Plot the confusion matrix as a heatmap with original class
names on both axes. Which pairs of classes are most often confused? Propose
a business-level explanation for the most frequent error (1–3 sentences).

**Task 4.3** - Per-class deep-dive: for the class with the lowest F1 score,
examine the misclassified samples in the **original** (un-preprocessed)
dataframe. Do their raw feature values resemble the true class or the
predicted class? What does this suggest about the ambiguity in the data
(2–3 sentences)?

---

## Part 5 - Hyperparameter Search

To use `NeuralNetwork` inside `RandomizedSearchCV`, you need to make it
sklearn-compatible.

**Task 5.1** - Add the following methods to your `NeuralNetwork` class in
`nn_multiclass_student.py` (do not modify `__init__` or any existing method):

```python
def get_params(self, deep=True):
    return {
        'layer_sizes':  self.layer_sizes,
        'activations':  self.activations,
        'random_state': self.random_state,
    }

def set_params(self, **params):
    for key, value in params.items():
        setattr(self, key, value)
    return self

def score(self, X, y):
    """Accuracy - used by sklearn's cross-validation."""
    return np.mean(self.predict(X) == y)
```

**Task 5.2** - Because `lr` and `n_epochs` are arguments of `fit()` rather
than `__init__`, sklearn's CV loop cannot set them via `set_params`. Wrap the
network in a thin `BaseEstimator` subclass:

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_sizes, activations,
                 lr=0.01, n_epochs=500, random_state=42):
        self.layer_sizes  = layer_sizes
        self.activations  = activations
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = NeuralNetwork(
            self.layer_sizes, self.activations, self.random_state
        )
        self.model_.fit(X, y, lr=self.lr,
                        n_epochs=self.n_epochs, verbose=False)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
```

**Task 5.3** - Run `RandomizedSearchCV` with ≥ 20 iterations, 3-fold CV,
`scoring='f1_macro'`, `random_state=42`, over the following space:

```python
param_distributions = {
    "layer_sizes": [
        [n_features, 32, 4],
        [n_features, 64, 4],
        [n_features, 128, 4],
        [n_features, 64, 32, 4],
        [n_features, 128, 64, 4],
        [n_features, 128, 64, 32, 4],
    ],
    "activations": [
        ['relu', 'softmax'],
        ['tanh', 'softmax'],
        ['relu', 'relu', 'softmax'],
        ['tanh', 'tanh', 'softmax'],
        ['relu', 'relu', 'relu', 'softmax'],
    ],
    "lr":       [0.0001, 0.001, 0.005, 0.01, 0.05],
    "n_epochs": [300, 500, 1000],
}
```

Note: the `activations` list length must always be `len(layer_sizes) - 1`.
The search space above is already consistent - do not mix entries from
different rows.

Print the best configuration found and its mean CV macro F1 score.

**Task 5.4** - Re-train `NNClassifier` with the best parameters on the full
training set. Report the final classification report and confusion matrix.
How much did macro F1 improve over the baseline from Part 3?
---

## Deliverables

- Completed Jupyter notebook with all tasks answered.
- Updated `nn_multiclass_student.py` with `get_params`, `set_params`, and
  `score` added (Task 5.1).
- All plots inline with axis labels, titles, and original class names.
- Written answers where requested (2–5 sentences each).

---
