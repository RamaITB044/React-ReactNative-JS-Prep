# Python Machine Learning Interview Questions and Answers

## Python Fundamentals

### 1. Explain the difference between lists and NumPy arrays. When would you use each?

**Answer:**

- **Lists:**

  - Dynamic size, can hold different data types
  - Slower for numerical operations
  - More memory intensive
  - Example:

  ```python
  my_list = [1, 2, 3, "hello", 4.5]
  ```

- **NumPy Arrays:**
  - Fixed size, homogeneous data type
  - Optimized for numerical operations
  - Memory efficient
  - Example:
  ```python
  import numpy as np
  my_array = np.array([1, 2, 3, 4, 5])
  # Fast operations
  result = my_array * 2  # Vectorized operation
  ```

**When to use each:**

- Use lists when:
  - Working with mixed data types
  - Need dynamic size
  - Simple data storage
- Use NumPy arrays when:
  - Performing numerical computations
  - Working with large datasets
  - Need vectorized operations
  - Machine learning algorithms

### 2. How do you handle missing data in pandas? Explain different methods.

**Answer:**

1. **Removing missing values:**

```python
import pandas as pd
# Remove rows with any missing values
df.dropna()
# Remove columns with any missing values
df.dropna(axis=1)
```

2. **Filling missing values:**

```python
# Fill with specific value
df.fillna(0)
# Fill with mean
df.fillna(df.mean())
# Forward fill
df.fillna(method='ffill')
# Backward fill
df.fillna(method='bfill')
```

3. **Interpolation:**

```python
# Linear interpolation
df.interpolate()
```

### 3. What are decorators in Python? Give an example of how they can be used in ML.

**Answer:**
Decorators are functions that modify the behavior of other functions.

**Example in ML:**

```python
import time
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper

@timer_decorator
def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)

# Usage
model = RandomForestClassifier()
train_model(model, X_train, y_train)
```

### 4. Explain the difference between shallow copy and deep copy in Python.

**Answer:**

- **Shallow Copy:** Creates a new object but references the same nested objects
- **Deep Copy:** Creates a new object and recursively copies all nested objects

**Example:**

```python
import copy

# Original list with nested structure
original = [[1, 2, 3], [4, 5, 6]]

# Shallow copy
shallow = copy.copy(original)
# or
shallow = original.copy()

# Deep copy
deep = copy.deepcopy(original)

# Modifying nested element
shallow[0][0] = 9  # Affects original
deep[0][0] = 9     # Doesn't affect original
```

### 5. How do you handle memory management in Python when working with large datasets?

**Answer:**

1. **Use generators instead of lists:**

```python
# Instead of
data = [x for x in range(1000000)]

# Use
def data_generator():
    for x in range(1000000):
        yield x
```

2. **Use appropriate data types:**

```python
import numpy as np
# Use smaller data types when possible
small_array = np.array([1, 2, 3], dtype=np.int8)  # 1 byte per element
```

3. **Clear unused variables:**

```python
import gc

# After processing large data
del large_dataframe
gc.collect()
```

4. **Use chunking for large files:**

```python
# Reading large CSV files
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

## Data Preprocessing

### 6. What are the different types of feature scaling? When would you use each?

**Answer:**

1. **StandardScaler (Z-score normalization):**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- Use when: Features follow normal distribution
- Formula: z = (x - μ) / σ

2. **MinMaxScaler:**

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

- Use when: Need values in specific range (e.g., [0,1])
- Formula: x_scaled = (x - x_min) / (x_max - x_min)

3. **RobustScaler:**

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

- Use when: Data contains outliers
- Uses median and quartiles instead of mean and std

### 7. Explain the concept of one-hot encoding and when to use it.

**Answer:**
One-hot encoding converts categorical variables into binary vectors.

**Example:**

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Original data
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red']
})

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(data[['color']])

# Result:
# red   = [1, 0, 0]
# blue  = [0, 1, 0]
# green = [0, 0, 1]
```

**When to use:**

- Categorical variables with no ordinal relationship
- When categorical variables are nominal
- When algorithms require numerical input

### 8. How do you handle categorical variables in your dataset?

**Answer:**

1. **One-hot encoding** (for nominal variables):

```python
pd.get_dummies(df['category'])
```

2. **Label encoding** (for ordinal variables):

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

3. **Target encoding** (for high cardinality):

```python
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'], df['target'])
```

4. **Hash encoding** (for very high cardinality):

```python
from category_encoders import HashingEncoder
encoder = HashingEncoder()
df_encoded = encoder.fit_transform(df['category'])
```

### 9. What is feature selection? Explain different methods.

**Answer:**

1. **Filter Methods:**

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
```

2. **Wrapper Methods:**

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
selector = RFE(LogisticRegression(), n_features_to_select=5)
X_selected = selector.fit_transform(X, y)
```

3. **Embedded Methods:**

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
# Features with non-zero coefficients are selected
selected_features = X.columns[lasso.coef_ != 0]
```

### 10. How do you detect and handle outliers in your data?

**Answer:**

1. **Statistical Methods:**

```python
# Z-score method
from scipy import stats
z_scores = stats.zscore(data)
outliers = (abs(z_scores) > 3)

# IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
```

2. **Visualization:**

```python
import seaborn as sns
# Box plot
sns.boxplot(data=df)

# Scatter plot
plt.scatter(x, y)
```

3. **Handling outliers:**

```python
# Remove outliers
df_clean = df[~outliers]

# Cap outliers
df['column'] = df['column'].clip(lower=lower_bound, upper=upper_bound)

# Replace with median
df['column'] = df['column'].mask(outliers, df['column'].median())
```

## Machine Learning Fundamentals

### 11. Explain the bias-variance tradeoff.

**Answer:**
The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between model complexity and prediction error.

**Example:**

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.legend()
    plt.show()
```

**High Bias (Underfitting):**

- Model is too simple
- High training error
- High test error
- Example: Linear regression for non-linear data

**High Variance (Overfitting):**

- Model is too complex
- Low training error
- High test error
- Example: Deep neural network with too many layers

### 12. What is cross-validation? Explain different types.

**Answer:**
Cross-validation is a technique to assess model performance by splitting data into training and validation sets.

1. **K-Fold Cross Validation:**

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

2. **Stratified K-Fold:**

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

3. **Leave-One-Out Cross Validation:**

```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

### 13. How do you handle imbalanced datasets?

**Answer:**

1. **Resampling Techniques:**

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversampling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Undersampling
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)
```

2. **Class Weights:**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

3. **Evaluation Metrics:**

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, _ = precision_recall_curve(y_true, y_pred)
ap = average_precision_score(y_true, y_pred)
```

### 14. Explain the difference between bagging and boosting.

**Answer:**
**Bagging (Bootstrap Aggregating):**

```python
from sklearn.ensemble import RandomForestClassifier
# Bagging example with Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

**Boosting:**

```python
from sklearn.ensemble import GradientBoostingClassifier
# Boosting example with Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X_train, y_train)
```

**Key Differences:**

- Bagging:

  - Parallel training
  - Independent models
  - Reduces variance
  - Example: Random Forest

- Boosting:
  - Sequential training
  - Dependent models
  - Reduces bias
  - Example: XGBoost, AdaBoost

### 15. What is regularization? Explain L1 and L2 regularization.

**Answer:**
Regularization prevents overfitting by adding penalty terms to the loss function.

**L1 Regularization (Lasso):**

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
# L1 tends to create sparse models
```

**L2 Regularization (Ridge):**

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
# L2 tends to shrink all coefficients
```

**Key Differences:**

- L1 (Lasso):

  - Adds absolute value of coefficients
  - Creates sparse models
  - Good for feature selection
  - Formula: Loss + λ∑|w|

- L2 (Ridge):
  - Adds squared value of coefficients
  - Shrinks all coefficients
  - Good for multicollinearity
  - Formula: Loss + λ∑w²

## Supervised Learning

### 16. Explain the working of Random Forest algorithm.

**Answer:**
Random Forest is an ensemble learning method that builds multiple decision trees.

**Example:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create a random forest classifier
rf = RandomForestClassifier(
    n_estimators=100,  # number of trees
    max_depth=10,      # maximum depth of trees
    random_state=42
)

# Train the model
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
```

**Key Components:**

1. Bootstrap sampling
2. Random feature selection
3. Decision tree construction
4. Voting/ averaging for prediction

**Advantages:**

- Handles non-linear relationships
- Reduces overfitting
- Handles missing values
- Provides feature importance

### 17. How does gradient boosting work? Explain XGBoost and LightGBM.

**Answer:**
Gradient Boosting builds models sequentially, each new model correcting errors of previous models.

**XGBoost Example:**

```python
import xgboost as xgb

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic'
}

# Train
model = xgb.train(params, dtrain, num_boost_round=100)
```

**LightGBM Example:**

```python
import lightgbm as lgb

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31
}

# Train
model = lgb.train(params, train_data, num_boost_round=100)
```

**Key Differences:**

- XGBoost:

  - Level-wise tree growth
  - More robust
  - Better for small datasets

- LightGBM:
  - Leaf-wise tree growth
  - Faster training
  - Better for large datasets

### 18. What is the difference between logistic regression and linear regression?

**Answer:**
**Linear Regression:**

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
# Predicts continuous values
```

**Logistic Regression:**

```python
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train, y_train)
# Predicts probabilities (0-1)
```

**Key Differences:**

1. Output:

   - Linear: Continuous values
   - Logistic: Probabilities (0-1)

2. Use Case:

   - Linear: Regression problems
   - Logistic: Classification problems

3. Function:
   - Linear: y = mx + b
   - Logistic: p = 1/(1 + e^(-z))

### 19. Explain the concept of decision trees and how they make decisions.

**Answer:**
Decision trees split data into subsets based on feature values.

**Example:**

```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Create and train tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=feature_names, class_names=target_names)
plt.show()
```

**Decision Process:**

1. Start at root node
2. Evaluate feature condition
3. Move to child node based on condition
4. Repeat until leaf node
5. Make prediction

**Splitting Criteria:**

- Gini impurity
- Information gain
- Entropy

### 20. How do you handle multicollinearity in your features?

**Answer:**

1. **Correlation Analysis:**

```python
import seaborn as sns
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
```

2. **VIF (Variance Inflation Factor):**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
```

3. **Feature Selection:**

```python
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X, y)
```

4. **PCA:**

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)
```

## Tips for Implementation

1. Always start with data exploration
2. Use appropriate evaluation metrics
3. Implement cross-validation
4. Monitor for overfitting
5. Document your preprocessing steps
