import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.LinearRegressionGD import LinearRegressionGD
from sklearn.linear_model import LinearRegression

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