import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl 
import pickle
import joblib
from sklearn import svm
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
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

filename = '***.csv'
data = pd.read_csv(filename)
X = data.iloc[:,4:]
X = X.fillna(0)
y = data.iloc[:,3]

c_best = 9
gamma_best = 0.01
seed = 42
# seed 29 37 49 51

X_train,X_test_verify,y_train,y_test_verify = train_test_split (
    X, y, random_state=seed, train_size = 0.8)
# 划分测试集和验证集
X_test,X_verify,y_test,y_verify = train_test_split (
    X_test_verify,y_test_verify, random_state =seed, train_size = 0.5)




for i in range(1,10,1):
    for j in range(1,10,1):
        j = j/100
        model = svm.SVC(C=i,kernel='rbf',gamma=j,decision_function_shape='ovo',probability=True)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test_verify)
        y_true = y_test_verify
        # F1-score
        f = f1_score(y_true, y_pred, average='binary')
        print('f1：',f,'c0:',i,'gamma0:',j)
        if f > f1:
            f1=f
            c_best=i
            gamma_best=j
print('f1：',f1,'c:',c_best,'gamma:',gamma_best)
start = time.perf_counter()

model = svm.SVC(C=c_best,kernel='rbf',gamma=gamma_best,decision_function_shape='ovo',probability=True) 
model.fit(X_train,y_train) 

# # 读取模型
# with open(modelname, 'rb') as f:
#     model = joblib.load(f)

print("数据集：", filename, "种子数：", seed)
# 训练集预测值，真值
y_pred = model.predict(X_test)
y_pred_proba = model.decision_function(X_test)
y_true = y_test

# 计算准确率
acc = accuracy_score(y_test, y_pred)
print(acc)
print(f1_score(y_true, y_pred,average='binary'))
# Recall
r = recall_score(y_true, y_pred, average='binary')
print(r)
# AUROC
fpr, tpr, th = roc_curve(y_true, y_pred_proba, pos_label=1)
print(auc(fpr, tpr))
precision, recall, _ = precision_recall_curve(y_true,y_pred_proba)
PRC = average_precision_score(y_true,y_pred_proba)
print(PRC)
pre = precision_score(y_test, y_pred)
print(pre)


end = time.perf_counter()
# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
# print("运行时间：", runTime, "秒")
print(runTime)
