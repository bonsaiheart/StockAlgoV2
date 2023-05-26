import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
predictions = pd.read_csv(''
                          'predictions.csv')
target = predictions["Target"]
predicted = predictions["Predictions"]

precision = precision_score(target, predicted)
accuracy = accuracy_score(target, predicted)
recall = recall_score(target, predicted)
f1 = f1_score(target, predicted)

print("Precision:", precision)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-Score:", f1)