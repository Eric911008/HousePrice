#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test 8
LinearRegression

@author: linweicheng
"""

import sklearn
import pandas as pd                                     #匯入檔案
from sklearn.model_selection import train_test_split    #將資料切割為訓練及測試集 
from sklearn.preprocessing import LabelEncoder          #把每個類別對應到某個整數
from sklearn.feature_selection import SelectKBest       #選出N個分數最高的特徵
from sklearn.feature_selection import f_regression      #適用於回歸模型的特徵評估 
from sklearn.linear_model import LinearRegression       #Logistic迴歸模型


train_data = pd.read_csv('train.csv')               #匯入檔案 
train_data = train_data.drop(['Id'], axis = 1)      #刪除Id欄位
x_data = train_data.drop(['SalePrice'], axis = 1)   #刪除SalePrice欄位
y_data = train_data['SalePrice']
print(x_data.shape, y_data.shape) #(1460, 79) (1460,)


x_data = x_data.dropna(axis = 1, how = 'any')       #處理缺失值（1是列，0是行），如果行或列中有任何空值，就會刪除它 
print(x_data.shape) #(1460, 60)

                                                   
encoder = LabelEncoder()                            #減少特徵數
                                                    #建立dataframe，針對column做改變
x_encoded = pd.DataFrame(x_data,columns=x_data.columns).apply(lambda col:encoder.fit_transform(col))
kbest = SelectKBest(f_regression, k=40)             #建立選取
x_new = kbest.fit_transform(x_encoded, y_data)
print(x_new.shape) #(1460, 40)

                                                    #切割資料，比例通常為 7:3 或 8:2
x_train, x_test, y_train, y_test = train_test_split(x_new, y_data, train_size=0.7, test_size=0.3)
model = LinearRegression()                          #建立模型

                                                    
model.fit(x_train, y_train)                         #預測結果及準確度
score = model.score(x_test, y_test)                  #比對預測結果和答案
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')

