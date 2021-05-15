from sklearn.linear_model import LinearRegression
import numpy as np


x = np.array([[5,1], [15,2], [25,6], [35,5], [45,2], [55,2]])
y = np.array([5, 20, 14, 32, 22, 38])
new_model = LinearRegression().fit(x, y)
print('intercept: ', new_model.intercept_)
print('coefficient: ', new_model.coef_)
###########################prediction#####################
y_pred1 = new_model.predict(x)
print(y_pred1)
print(x)
y_pred = new_model.intercept_ + np.sum(new_model.coef_ * x, axis=1)
print('predicted response:', y_pred, sep='\n')