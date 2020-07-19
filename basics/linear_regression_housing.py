import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.LinearRegressionGD import LinearRegressionGD
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')    
    return None

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], height=2)
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

X = df[['RM']].values
y = df['MEDV'].values

sc_X = StandardScaler()
sc_y = StandardScaler()

X_std = sc_X.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1, 1))
y_std = y_std.reshape(1, X_std.shape[0])[0]

lr = LinearRegressionGD()
lr.fit(X_std, y_std)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

num_rooms_std = sc_X.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print(f'Price in $1000s with GD: {sc_y.inverse_transform(price_std)[0]:.3f}')
lin_regplot(X_std, y_std, lr)
plt.title('Linear Regression Model with GD')
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

slr = LinearRegression()
slr.fit(X, y)
print(f'Price in $1000s with Sklearn: {slr.predict([[5.0]])[0]:.3f}')
lin_regplot(X, y, slr)
plt.title('Linear Regression Model with Sklearn')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()

# removing outliers with RANdom SAmple Consensus (RANSAC)

ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_threshold=5.0, random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

# evaluating the performance of the linear regression model

X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

# for a good regression model, we would expect that the errors are randomly distributed and the residuals should be randomly scattered around the centerline.

# MSE = 1/n*(sum(y[i] - ˆy[i])**2)
print(f'MSE train: {mean_squared_error(y_train, y_train_pred):.3f}, test: {mean_squared_error(y_test, y_test_pred):.3f}')

# Rˆ2 = 1 - MSE/var(y)
print(f'Rˆ2 train: {r2_score(y_train, y_train_pred):.3f}, test: {r2_score(y_test, y_test_pred):.3f}')

# regularized methods for regression

# j(w)ridge = sum(y[i] - ˆy[i])**2 + alpha*||w||2
# l2 = ||w||2 = alpha*sum(w[i]**2)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

print(f'Rˆ2 train with Ridge regularizer: {r2_score(y_train, y_train_pred):.3f}, test: {r2_score(y_test, y_test_pred):.3f}')

# j(w)lasso = sum(y[i] - ˆy[i])**2 + alpha*||w||1
# l1 = ||w||1 = alpha*sum(abs(w[i]))

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print(f'Rˆ2 train with Lasso regularizer: {r2_score(y_train, y_train_pred):.3f}, test: {r2_score(y_test, y_test_pred):.3f}')

# j(w)elasticnet = sum(y[i] - ˆy[i])**2 + alpha*||w||2 + alpha*||w||1

elastic_net = ElasticNet(alpha=1.0)
elastic_net.fit(X_train, y_train)
y_train_pred = elastic_net.predict(X_train)
y_test_pred = elastic_net.predict(X_test)

print(f'Rˆ2 train with Elastic Net regularizer: {r2_score(y_train, y_train_pred):.3f}, test: {r2_score(y_test, y_test_pred):.3f}')
