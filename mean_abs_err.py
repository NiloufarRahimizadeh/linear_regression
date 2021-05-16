from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
e1 = mean_absolute_error(y_true, y_pred)
print(e1)

e2 = mean_squared_error(y_true, y_pred)
print(e2)