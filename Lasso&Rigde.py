#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
活動22
Lasso,Rigde

@author: linweicheng
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

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

mean_rmse = []
for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha, solver="cholesky")
    rmse_list = np.sqrt(-cross_val_score(ridge_reg, X, y
                                         , scoring="neg_mean_squared_error", cv=3))
    mean_rmse.append(rmse_list.mean())
    
cv_ridge = pd.Series(mean_rmse, index=alphas)
print(cv_ridge.idxmin())


ridge_reg = Ridge(alpha=3, solver="cholesky")
ridge_reg.fit(X_train, y_train)

y_pred_train = ridge_reg.predict(X_train)
y_pred_test = ridge_reg.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(train_rmse)
print(test_rmse)

ridge_reg.fit(X_train, y_train)                         
score1 =ridge_reg.score(X_test, y_test)




alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

mean_rmse = []
for alpha in alphas:
    ridge_reg = Lasso(alpha=alpha)
    rmse_list = np.sqrt(-cross_val_score(ridge_reg, X, y
                                         , scoring="neg_mean_squared_error", cv=3))
    mean_rmse.append(rmse_list.mean())
    
cv_ridge = pd.Series(mean_rmse, index=alphas)
print(cv_ridge.idxmin())


ridge_reg = Lasso(alpha=3)
ridge_reg.fit(X_train, y_train)

y_pred_train = ridge_reg.predict(X_train)
y_pred_test = ridge_reg.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(train_rmse)
print(test_rmse)

ridge_reg.fit(X_train, y_train)                         
score2 =ridge_reg.score(X_test, y_test)   
print("__"*23)   
print("Ridge :")                  
print('Score: ', score1)
print('Accuracy: ' + str(score1*100) + '%')
print("--"*23)            
print("Lasso :")
print('Score: ', score2)
print('Accuracy: ' + str(score2*100) + '%')
print("__"*23) 
