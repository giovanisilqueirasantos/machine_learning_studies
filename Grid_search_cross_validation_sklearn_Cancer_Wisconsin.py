import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

pipe_svc = Pipeline([('sc1', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = [{
    'clf__C':  param_range,
    'clf__kernel': ['linear']
}, {
    'clf__C':  param_range,
    'clf__gamma':  param_range,
    'clf__kernel': ['rbf']
}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_, gs.best_params_)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')

# Nested cross-validation with SVC

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
print(f'Nested CV SVC accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# Nested cross-validation with Decision tree

param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=param_grid, scoring='accuracy', cv=5)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print(f'Nested CV Decision tree accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# SVC accuracy on test data > Decision tree accuracy