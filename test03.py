from sklearn.linear_model import LinearRegression
import numpy as np


x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])
new_model = LinearRegression().fit(x.reshape((-1, 1)), y)
print('intercept: ', new_model.intercept_)
print('coefficient: ', new_model.coef_)
