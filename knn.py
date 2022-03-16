#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
活動24

@author: linweicheng
"""

import pandas as pd #匯入檔案
from sklearn.model_selection import train_test_split #將資料切割為訓練及測試集 
from sklearn.preprocessing import LabelEncoder #把每個類別對應到某個整數
from sklearn.feature_selection import SelectKBest #選出K個分數最高的特徵
from sklearn.feature_selection import f_regression #適用於回歸模型的特徵評估 
from sklearn.linear_model import LinearRegression #Logistic迴歸模型

train_data = pd.read_csv('train.csv')               
train_data = train_data.drop(['Id'], axis = 1)     
x_data = train_data.drop(['SalePrice'], axis = 1)  
y_data = train_data['SalePrice']
print(x_data.shape, y_data.shape) 
x_data = x_data.dropna(axis = 1, how = 'any')      
print(x_data.shape) 

                                                   
encoder = LabelEncoder()                            
                                                   
x_encoded = pd.DataFrame(x_data,columns=x_data.columns).apply(lambda col:encoder.fit_transform(col))
kbest = SelectKBest(f_regression, k = 40 )            
X_new = kbest.fit_transform(x_encoded, y_data)
print(X_new.shape) 

                                                
              
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X_new, y_data , train_size=0.7, test_size=0.3, random_state=0)

model_knn = KNeighborsRegressor(n_neighbors = 1 , p = 2)
model_knn.fit(X_train,y_train)
score = model_knn.score(X_test, y_test)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')
 


