from sklearn.metrics import matthews_corrcoef


y_true = [+1, +1, +1, +1]
y_pred = [+1, +1, +1, +1]
out = matthews_corrcoef(y_true, y_pred)
print(out)