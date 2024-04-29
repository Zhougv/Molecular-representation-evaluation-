import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import time
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score,f1_score,recall_score,precision_score
f=open("***.csv")
df=pd.read_csv(f)

X=df.iloc[:,0:1024]
y=df.iloc[:,1024:]

X_train,X_test_verify,y_train,y_test_verify = train_test_split (
    X, y, random_state=51, train_size = 0.8)
X_test,X_verify,y_test,y_verify = train_test_split (
    X_test_verify,y_test_verify, random_state =51, train_size = 0.5)

ytest= np.array(y_test)
yverify= np.array(y_verify)

A=[]
B=[]
D=[]
E=[]
Z=[]
varr = 0

for max_depth in range(3,9,1):
   for n_estimators in range(30,1000,10):
       xgb_model=xgb.XGBClassifier(max_depth=max_depth,
                                      n_estimators=n_estimators,
                                      learning_rate=0.05,
                                      objective="reg:squarederror",
                                      booster="gbtree",
                                      random_state=0
                                      )
       xgb_model.fit(X_train,y_train)
       y_predic=xgb_model.predict(X_verify)
       y_pred_proba=xgb_model.predict_proba(X_verify)
       yy=y_pred_proba[:, 1]
       fpr, tpr, thresholds = metrics.roc_curve(yverify, y_pred_proba[:, 1], pos_label=1)
       roc_auc = metrics.auc(fpr, tpr)
       varr+=1
       A.append(max_depth)
       B.append(n_estimators)
       D.append(roc_auc)
       E.append(varr)
       print(varr)

location=D.index(max(D))
bestestimator=B[location]
bestmaxdepth=A[location]
value=max(D)
print("验证集最优n_estimators是"+str(B[location]),"验证集最优maxdepth是"+str(A[location]))
print("验证集最优AUC"+str(D[location]))

xgb_modelfinal= xgb.XGBClassifier(max_depth=bestmaxdepth,
                              n_estimators=bestestimator,  # 0.419,
                              learning_rate=0.05,  # 400,
                              objective="reg:squarederror",
                              booster="gbtree",
                              random_state=0
                              )
start=time.time()
xgb_modelfinal.fit(X_train,y_train)
end=time.time()
y_predicfinal = xgb_modelfinal.predict(X_test)
y_pred_probafinal= xgb_modelfinal.predict_proba(X_test)
fprfinal, tprfinal, thresholds = metrics.roc_curve(ytest, y_pred_probafinal[:, 1], pos_label=1)
roc_aucfinal = metrics.auc(fprfinal, tprfinal)
AUPR = average_precision_score(ytest, y_pred_probafinal[:, 1])
ACC = accuracy_score(ytest, y_predicfinal)
F1 = f1_score(ytest, y_predicfinal)
Pre = precision_score(ytest, y_predicfinal)
Recall = recall_score(ytest, y_predicfinal)

Z.append(roc_aucfinal)
locationfinal=Z.index(max(Z))

print("测试集最优AUC"+str(Z[locationfinal]))
print("测试集最优AUPR")
print(AUPR)
print("测试集最优ACC")
print(ACC)
print("测试集最优F1")
print(F1)
print("测试集最优Pre")
print(Pre)
print("测试集最优Recall")
print(Recall)
print("XGBoost程序的运行时间为：{}".format(end-start))
