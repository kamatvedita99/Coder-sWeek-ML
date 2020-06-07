from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle

dataset = pd.read_csv('./Social_Network_Ads.csv', delimiter=",")

dataset = dataset.drop('User ID',axis=1)

dataset['Gender'] = dataset['Gender'].astype('category')
dataset['Gender'] = dataset['Gender'].cat.codes

Y = dataset.iloc[:,-1:].values
X = dataset.iloc[:,0:3].values

seed = 9
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

pickle.dump(model,open('model.pkl','wb'))