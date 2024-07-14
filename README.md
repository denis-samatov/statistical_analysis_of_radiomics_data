# Data Analysis and Model Building

This repository contains Python code for analyzing medical data, specifically radiomics features, and building machine learning models for classification tasks.

## Overview

The project involves multiple stages including data filtering, handling missing values, statistical analysis, feature selection, and model building. Visualizations and statistical tests are used to enhance understanding and accuracy. A key tool used in this project is the [Feature-Selector](https://github.com/WillKoehrsen/feature-selector.git) repository, which aids in selecting the most informative features.

## Library Installation

Necessary libraries are installed, and modules are imported to ensure smooth workflow execution.

```python
!git clone https://github.com/WillKoehrsen/feature-selector.git
!pip install lightgbm tabgan
!pip install lightgbm --upgrade

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
```

## Data Filtering

Data is loaded, specific columns are removed, and the dataset is split into two parts: "bad" and "good" columns.

```python
# Load data
df = pd.read_excel('/content/medical_data.xlsx')
df_copy = df.copy()
df_copy.drop(columns=['ВК'], inplace=True)

# Split data into 'bad' and 'good' columns
bad_columns = df_copy.columns[:df_copy.shape[1]//2+1]
good_columns = df_copy.columns[df_copy.shape[1]//2+1:]
df_bad = df_copy[bad_columns]
df_good = df_copy[good_columns]
df_good[df_copy.columns[0]] = df_copy.iloc[:, 0]
df_good = df_good.reindex(columns=[df_copy.columns[0], *good_columns])
```

## Handling Missing Values

Missing values are visualized, columns and rows with a high percentage of missing data are removed, and remaining missing values are filled with the mean.

```python
sns.heatmap(df_copy.isnull(), cbar=False, yticklabels=False)

# Remove columns and rows with too many missing values
threshold = 0.75
df_copy = df_copy.dropna(thresh=threshold * df_copy.shape[0], axis=1)
df_copy = df_copy.dropna(thresh=threshold * df_copy.shape[1], axis=0)

# Fill remaining missing values with mean
df_copy = df_copy.fillna(df_copy.mean())

sns.heatmap(df_copy.isnull(), cbar=False, yticklabels=False)
```

## Statistical Analysis

Descriptive statistics for "bad" and "good" columns are calculated to understand the data distribution.

```python
df_bad.describe()
df_good.describe()
```

## U-Test

A U-test is conducted to compare the distributions of "bad" and "good" data columns.

```python
def u_test(param, data_1, data_2):
    result = stats.mannwhitneyu(data_1, data_2, alternative='two-sided')
    return {'param': param, 'statistic': result[0], 'pvalue': result[1]}

results = [u_test(param, df_bad[param], df_good[param]) for param in df_bad.columns]
df_u_test = pd.DataFrame(results)
df_u_test['hypothesis'] = np.where(df_u_test['pvalue'] > 0.05, 'H0', 'H1')
df_u_test.groupby('hypothesis').agg('count')
```

## Feature Selection using Feature-Selector

Feature-Selector is used to identify the most informative features by detecting missing values and collinear features.

```python
# Identify missing data and collinear features
fs = FeatureSelector(data=df_bad, labels=np.zeros(df_bad.shape[0]))
fs.identify_missing(missing_threshold=0.6)
fs.identify_collinear(correlation_threshold=0.98)
```

### Collinearity Diagrams

The *identify_collinear* method finds collinear predictors based on a specified correlation threshold.

<div align="center">
  <img src="https://github.com/denis-samatov/statistical_analysis_of_radiomics_data/blob/main/img_1.png" alt="Collinearity Diagram">
</div>

### Zero Importance Features

The *identify_zero_importance* method finds features with zero importance. These features are identified using a LightGBM model.

<div align="center">
  <img src="https://github.com/denis-samatov/statistical_analysis_of_radiomics_data/blob/main/img_2.png" alt="Normalized Feature Importances">
</div>

### Cumulative Importance of Features

The cumulative importance of features is plotted to identify a threshold for selecting the most important features.

<div align="center">
  <img src="https://github.com/denis-samatov/statistical_analysis_of_radiomics_data/blob/main/img_3.png" alt="Cumulative Feature Importance">
</div>

### Features with Single Unique Value

This method identifies columns containing only one unique value.

<div align="center">
  <img src="https://github.com/denis-samatov/statistical_analysis_of_radiomics_data/blob/main/img_4.png" alt="Number of Unique Values per Feature">
</div>

## Removing Unnecessary Features

Unnecessary features identified by Feature-Selector are removed from the dataset.

```python
df_informative_features = fs.remove(methods='all', keep_one_hot=False)
```

## Data Analysis Methods and Model Building

Principal Component Analysis (PCA) and Lasso are used for feature selection, followed by Logistic Regression and K-Nearest Neighbors (KNN) for model building.

### PCA for Feature Selection

```python
X_centered = X - X.mean()
cov_matrix = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
index_informative_features = np.where(eigenvalues > np.median(eigenvalues))[0]
df_informative_features = X.iloc[:, list(index_informative_features)]
```

### Lasso for Feature Selection

```python
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X, y)
index_informative_features = np.where(clf.coef_ != 0)[0]
df_informative_features = X.iloc[:, list(index_informative_features)]
```

### Logistic Regression

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LogisticRegression(penalty='l1', solver='liblinear'))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### K-Nearest Neighbors (KNN)

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

<div align="center">
  <img src="https://github.com/denis-samatov/statistical_analysis_of_radiomics_data/blob/main/img_5.png" alt="ROC Curve and AUC">
</div>
