import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
import time
import joblib

from math import sqrt
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# 导入数据
filename = '***.csv'
data = pd.read_csv(filename)
# 划分数据与标签：x为数据，y为标签
X = data.iloc[:,5:]
y = data.iloc[:,3]


X_train,X_test_verify,y_train,y_test_verify = train_test_split (
    X, y, random_state= seed, train_size = 0.8)
X_test,X_verify,y_test,y_verify = train_test_split (
    X_test_verify,y_test_verify, random_state = seed, train_size = 0.5)


for i in range(1,10,4):
    for j in range(1,10,4):
        j = j/100
        model = svm.SVR(C=i, kernel='rbf', gamma=j)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_verify)
        # RMSE
        RMSE = sqrt(mean_squared_error(y_verify,y_pred))
        print('RMSE：',RMSE,'c0:',i,'gamma0:',j)
        if RMSE < RMSE0:
            RMSE0=RMSE
            c_best=i
            gamma_best=j
print('RMSE：',RMSE0,'c:',c_best,'gamma:',gamma_best)

# 代码开始时间
start = time.perf_counter()

model = svm.SVR(C=c_best,kernel='rbf',gamma=gamma_best)
model.fit(X_train,y_train) 

# with open(modelname, 'rb') as f:
#         model = joblib.load(f)

# 训练集预测值，真值
y_pred = model.predict(X_test)
y_true = y_test


#计算皮尔森相关系数、R2、RMSE
RMSE = sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
PCC = pearsonr(y_test,y_pred)
R2 = r2_score(y_test,y_pred)
Spearman = spearmanr(y_test,y_pred)

print("数据集：", filename, "种子数：", seed)

print(RMSE)
print(MAE)
print(PCC)
print(R2)
print(Spearman)

# 获取结束时间
end = time.perf_counter()
# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
# print("运行时间：", runTime, "秒")
print(runTime)
