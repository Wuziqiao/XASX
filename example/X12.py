import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import warnings

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
excelFile = r'E:\PycharmProject\XASXG_model1\datasets\datasets\D4.xlsx'


print("**************/seasonal_decompose/***************************")
data = pd.read_excel(excelFile, index_col = [0])
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['E'],model='multiplicative',extrapolate_trend='freq')
trend = decomposition.trend
seasonal = decomposition.seasonal
resid = decomposition.resid
data.insert(0,'trend',trend)
data.insert(0,'seasonal',seasonal)
data.insert(0,'resid',resid)
E = data[['E']].values

print("**************/ARIMA/***************************")

p = 3
q = 4

model = sm.tsa.arima.ARIMA(trend, order = (p, 1, q))
result = model.fit()
x = result.predict(start='2016/3/1', end= '2021/12/1')
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(trend[-70:], x.values))
x = x.values
for i in range(2):
    x =np.insert(x,0,-1,axis=0)
data.insert(0,'tcf',x)

# # print(data.columns)
"""
    SVR module
"""

import time
from sklearn.svm import SVR

pjme = data.iloc[3:,:]

def spliteData(m):
    if m >= 10:
        split_date = '2021-' + str(m) + '-01'  # M指的是预测的月份
    else:
        split_date = '2021-0' + str(m) + '-01'  # M指的是预测的月份
    pjme_train = pjme.loc[pjme.index < split_date].copy()
    pjme_test = pjme[pjme['year'].isin([2021])]
    pjme_test = pjme_test[pjme_test['month'].isin([m])]
    return pjme_train,pjme_test
def create_features(df, feature,label=None):
    """
    Creates atime series features from datetime index
    """
    df['date'] = df.index

    X = df[feature]

    if label:
        y = df[label]
        return X, y
    return X

mape=[]
preLt = np.array([])
print("**************/predict Lt/***************************")
for i in range(1,13):
    pjme_train, pjme_test = spliteData(i)
    ptest = pjme_test.copy()
    ptrain = pjme_train.copy()
    feature = ['ds', 'T', 'month', 'tcf']
    X_train, y_train = create_features(pjme_train, feature,label='trend')
    X_test, y_tes = create_features(pjme_test, feature,label='trend')
    y = y_train.copy()
    Y = y_tes.copy()

    X_test = np.array(X_test)
    X_train = np.array(X_train)
    y_test = np.array(y_tes)
    y_train = np.array(y_train)

    linear_svr = SVR(kernel='linear')  # 线性核函数初始化的SVR
    linear_svr.fit(X_train, y_train)
    linear_svr_y_predict = linear_svr.predict(X_test)
    preLt = np.insert(preLt,i-1,linear_svr_y_predict[0],axis=0)

"""
    XGBoost module
"""
print("**************/predict Li Ls/***************************")
import xgboost as xgb
pjme = data.copy()
preLi = np.array([])
preLs = np.array([])
for i in range(1,13):
    pjme_train, pjme_test = spliteData(i)
    ptest = pjme_test.copy()
    ptrain = pjme_train.copy()

    feature = ['ds', 'T']
    X_train_1, y_train_1 = create_features(pjme_train,feature, label='resid')#预测ir
    X_test_1, y_test_1 = create_features(pjme_test, feature,label='resid')
    # feature = ['ds', 'month']
    # X_train_2, y_train_2 = create_features(pjme_train, feature, label='seasonal')  # 预测ir
    # X_test_2, y_test_2 = create_features(pjme_test, feature, label='seasonal')
    lr = 0.11
    n_esm = 1000
    max_depth = 2
    """
    predict Li
    """
    reg1 = xgb.XGBRegressor(
          n_estimators=n_esm, learning_rate=lr, min_child_weight=1,max_depth=max_depth)
    # dtrain = xgb.DMatrix(X_train_1,label=y_train_1)
    reg1.fit(X_train_1, y_train_1,
            eval_set=[(X_train_1, y_train_1), (X_test_1, y_test_1)],
            early_stopping_rounds=100,
           verbose=False) # Change verbose to True if you want to see it train
    yLi = reg1.predict(X_test_1)
    preLi = np.insert(preLi,i-1,yLi[0],axis=0)

    """
    predict Ls
    # """
    reg2 = xgb.XGBRegressor(
        n_estimators=n_esm, learning_rate=lr, min_child_weight=1, max_depth=max_depth)
    # dtrain = xgb.DMatrix(X_train_1, label=y_train_1)
    # reg2.fit(X_train_2, y_train_2,
    #          eval_set=[(X_train_2, y_train_2), (X_test_2, y_test_2)],
    #          early_stopping_rounds=100,
    #          verbose=False)  # Change verbose to True if you want to see it train
    # yLs = reg2.predict(X_test_2)
    # preLs = np.insert(preLs,i-1,yLs[0],axis=0)
preLs = data['seasonal'].values
preLs = preLs[-24:-12]
ypre = np.array([])

for i in range(12):
    ypre = np.insert(ypre,i,preLs[i]*preLi[i]*preLt[i], axis= 0)
preY = pd.DataFrame(columns=['result'], data=ypre)
print(mean_absolute_percentage_error(E[-12:],preY))
# preY.to_excel('result.xlsx')

