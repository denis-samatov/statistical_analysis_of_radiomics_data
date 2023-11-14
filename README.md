# Анализ данных и построение моделей

Этот репозиторий содержит код на языке Python, который выполняет анализ медицинских данных и строит модели машинного обучения для задачи классификации.

## Загрузка библиотек

В данном блоке кода происходит установка необходимых библиотек и импорт модулей для дальнейшей работы.

```python
!git clone https://github.com/WillKoehrsen/feature-selector.git
!pip install lightgbm
!pip install tabgan
!pip install lightgbm --upgrade

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import tabgan
import lightgbm as lgb

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
```

## Фильтрация данных

В этом блоке кода осуществляется фильтрация данных, удаление выбранных столбцов и разделение данных на две части: "хорошие" и "плохие" столбцы.

```python
# Загрузка данных из файла
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_excel('/content/medical_data.xlsx')

# Удаление выбранного столбца 'ВК'
df_copy = df.copy()
df_copy.drop(columns=['ВК'], inplace=True)

# Разделение данных на две части: 'хорошие' и 'плохие' столбцы
bad_columns = df_copy.columns[:df_copy.shape[1]//2+1]
good_columns = df_copy.columns[df_copy.shape[1]//2+1:]

df_bad = df_copy[bad_columns]
df_good = df_copy[good_columns]

# Преобразование данных для визуализации
df_good[str(df_copy.columns[0])] = df_copy.iloc[:, 0]
df_good = df_good.reindex(columns=[str(df_copy.columns[0]), *list(good_columns)])
```

## Обработка пропущенных значений

Этот блок кода отвечает за обработку пропущенных значений в данных. Пропуски заполняются средними значениями, а также производится удаление столбцов и строк с пропущенными значениями свыше порогового значения.

```python
# Визуализация пропущенных значений
sns.heatmap(df_copy.isnull(), cbar=False, yticklabels=False);

# Удаление столбцов и строк с пропущенными значениями
threshold = 0.75
df_copy = df_copy.dropna(thresh=threshold * df_copy.shape[0], axis=1)
df_copy = df_copy.dropna(thresh=threshold * df_copy.shape[1], axis=0)

# Заполнение пропущенных значений средними
df_copy = df_copy.fillna(df_copy.mean())

# Визуализация после обработки
sns.heatmap(df_copy.isnull(), cbar=False, yticklabels=False);
```

## Статистический анализ

В этом блоке кода проводится статистический анализ данных. Сначала выводятся описательные статистики для "плохих" и "хороших" данных.

```python
# Описательная статистика для 'плохих' данных
df_bad.describe()

# Описательная статистика для 'хороших' данных
df_good.describe()
```

## U-тест

В данном блоке кода проводится U-тест для сравнения распределений "плохих" и "хороших" данных по каждому параметру.

```python
# Функция для проведения U-теста
def u_test(parametr, data_1, data_2):
    result = stats.mannwhitneyu(list(data_1), list(data_2), alternative='two-sided')
    return {'parametr': parametr,
            'statistic': result[0],
            'pvalue': result[1]
            }

# Применение U-теста ко всем параметрам
numpy_bad = df_bad.to_numpy()[:, 1:]
numpy_good = df_good.to_numpy()[:, 1:]

result = list(map(u_test, df_bad.columns[1:], numpy_bad.T, numpy_good.T))

# Создание DataFrame с результатами U-теста
df_u_test = pd.DataFrame(result)

# Разделение результатов на гипотезы H0 и H1
df_u_test['hypothesis'] = np.where(df_u_test['pvalue'] > 0.05, 'H0', 'H1')

# Группировка результатов
df_u_test.groupby(['hypothesis'], group_keys=True).agg(['count'])
```

## Выбор информативных признаков с использованием Feature-Selector

В этом блоке кода применяется библиотека Feature-Selector для выбора наиболее информативных признаков.

```python
# Идентификация отсутствующих данных и коллинеарных признаков для 'плохих' данных
fs = FeatureSelector(data=df_bad, labels=np.zeros(df_bad.shape[0]))
fs.identify_missing(missing_threshold=0.6)
fs.identify_collinear(correlation_threshold=0.98)

# Идентификация отсутствующих данных и коллинеарных признаков для 'хороших' данных
fs = FeatureSelector(data=df_good, labels=np.ones(df_good.shape[0]))
fs.identify_missing(missing_threshold=0.6)
fs.identify_collinear(correlation_threshold=0.98)

# Идентификация признаков для всего датасета
df_bad_copy = df_bad.copy()
df_good_copy = df_good.copy()

df_good_copy.columns = list(df_bad_copy.columns)

df_bad_copy['label'] = np.zeros(df_bad_copy.shape[0])
df_good_copy['label'] = np.ones(df_good_copy.shape[0])

new_df = pd.concat([df_bad_copy, df_good_copy], ignore_index=True)

new_df =

 new_df.sample(frac=1).reset_index(drop=True)

X = new_df.iloc[:, 1:-1]
X = (X - X.mean()) / X.std()

y = new_df['label']

fs = FeatureSelector(data=X, labels=y)

# Идентификация пропущенных данных, коллинеарных и низкозначимых признаков
fs.identify_all(selection_params={'missing_threshold': 0.6,
                                  'correlation_threshold': 0.98,
                                  'task': 'classification',
                                  'eval_metric': 'auc',
                                  'importance_type': 'split',
                                  'cumulative_importance': 0.99})
```

## Удаление ненужных признаков

В этом блоке кода происходит удаление признаков, выявленных различными методами Feature-Selector.

```python
# Удаление ненужных признаков
df_informative_features = fs.remove(methods='all', keep_one_hot=False)
```

## Применение методов анализа данных и построение моделей

Здесь представлен код, который применяет методы анализа данных (PCA, Lasso) и строит модели машинного обучения (логистическая регрессия, метод ближайших соседей).

```python
# PCA для выбора признаков
X_centered = X - X.mean()
cov_matrix = np.cov(X_centered)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
index_informative_features = np.where(eigenvalues > np.median(eigenvalues))[0]
df_informative_features = X.iloc[:, list(index_informative_features)]

# Lasso для выбора признаков
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X, y)
index_informative_features = np.where(clf.coef_ != 0)[0]
df_informative_features = X.iloc[:, list(index_informative_features)]

# Логистическая регрессия
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Масштабирование признаков
    ('lasso', LogisticRegression(penalty='l1', solver='liblinear'))  # Логистическая регрессия с L1-регуляризацией
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Масштабирование признаков
    ('knn', KNeighborsClassifier(n_neighbors=3))  # K-ближайших соседей
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

Этот код представляет собой комплексный анализ медицинских данных, включая фильтрацию, обработку пропущенных значений, статистический анализ, выбор признаков и построение моделей. Результаты представлены в виде визуализаций, статистических таблиц и оценок моделей.
