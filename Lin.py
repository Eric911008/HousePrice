"""
Linaer Regression

"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

df = pd.read_csv("house_price.csv",nrows=1000)
df.dropna(inplace=True)

y = df["Price"]
X = df.drop(columns=["Price"])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(train_rmse)
print(test_rmse)

lin_reg.fit(X_train, y_train)                         #預測結果及準確度
score =lin_reg.score(X_test, y_test)                  #比對預測結果和答案
print("Linaer Regression :")
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')
print("__"*20)
